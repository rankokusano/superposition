# v15 (D) NS CPU checks (docs/v15_experiment_log.md Sec.33.5). Run from preference_inference inside the container, CPU only:
#   CUDA_VISIBLE_DEVICES= python analyze/v15/check_ns_v15.py
import os, sys, types
import numpy as np, torch, h5py
sys.path.insert(0, os.getcwd())
import util
from model import model_v15, model_v15ns
from analyze.v15 import eval_s2_v15 as E
torch.set_num_threads(2)
RES = 'data/result'
def cfg(exp):
    c = util.load_config(f'config/exp/{exp}.yml'); m = util.load_config(f'config/model/{c.model.name}/{c.model.config}.yml'); return c, m
def build(exp, seed, ns_mod):
    c, m = cfg(exp); util.seed_all(seed)
    cls = getattr(model_v15ns if ns_mod else model_v15, c.model.name)
    net = cls(m)
    enc = c.train.get('pretrain_encoder_from', None)
    if enc:
        ck = torch.load(enc, map_location='cpu')['model']
        keep = {k: v for k, v in ck.items() if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module', 'share_lns'))}
        r = net.load_state_dict(keep, strict=False); assert not r.unexpected_keys
    return c, net
def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)
print('== 1. phase-1 init: NS vs shared (same seed), and vs the saved shared ep0 checkpoint')
for cond in ('exp', 'noexp'):
    for s in (0, 1, 2):
        _, sh = build(f'v15_s2p1_{cond}_v15i_mse', s, False); _, ns = build(f'v15_s2p1_{cond}_v15i_ns_mse', s, True)
        a, b = sh.state_dict(), ns.state_dict()
        extra = sorted(set(b) - set(a)); miss = sorted(set(a) - set(b))
        ok_common = same(a, {k: b[k] for k in a})
        ok_copy = all(torch.equal(b[k], b[k.replace('lstm_other', 'lstm')]) for k in extra)
        p = os.path.join(RES, f'v15_s2p1_{cond}_v15i_mse', str(s), 'model', '00000.pth')
        ck = torch.load(p, map_location='cpu')['model'] if os.path.exists(p) else None
        ok_ck = None if ck is None else (set(ck) == set(a) and same({k: b[k] for k in a}, ck))
        ntr = lambda n: sum(q.numel() for q in n.parameters())
        print(cond, s, 'common keys equal', ok_common, '| extra', extra, '| missing', miss, '| lstm_other == lstm', ok_copy,
              '| NS common == saved shared ep0', ok_ck, '| params', ntr(sh), '->', ntr(ns))
print('== 2. phase-1 freezing (trainable superposition keys)')
c, ns = build('v15_s2p1_exp_v15i_ns_mse', 0, True)
util.gen_optim_params(ns, c.train.optim_config) if False else None
fr = c.train.optim_config.freeze
tr = [n for n, _ in ns.named_parameters() if not any(f in n for f in fr)]
print('phase 1 trainable superposition:', [n for n in tr if 'superposition' in n])
print('== 3. phase-2 class: init vs shared phase-2 (same seed), pretrain hand-over, freezing')
for s in (0, 1):
    c2, m2 = cfg(f'v15_s2p2h_exp_v15i_ns' + ('' if s == 0 else f'_s{s}'))
    util.seed_all(s); sh2 = model_v15.SuperpositionNetworkMotionPleasureHoldOtherV15(m2)
    util.seed_all(s); ns2 = model_v15ns.SuperpositionNetworkMotionPleasureHoldOtherV15NS(m2)
    a, b = sh2.state_dict(), ns2.state_dict()
    print('seed', s, 'phase-2 common keys equal', same(a, {k: b[k] for k in a}), 'extra', sorted(set(b) - set(a)))
    # hand-over: perturb the NS phase-1 weights so lstm and lstm_other differ, then load as train_v12 does (strict False)
    _, p1 = build('v15_s2p1_exp_v15i_ns_mse', s, True)
    with torch.no_grad():
        for n, q in p1.named_parameters():
            if 'lstm_other' in n: q.add_(0.123)
    sd = p1.state_dict()
    r = ns2.load_state_dict(sd, strict=False)
    print('  missing', sorted({k.split('.')[0] for k in r.missing_keys}), 'unexpected', r.unexpected_keys,
          '| lstm_other loaded', all(torch.equal(ns2.state_dict()[k], sd[k]) for k in sd if 'lstm_other' in k),
          '| lstm != lstm_other after load', not torch.equal(ns2.superposition_module.lstm.weight_ih, ns2.superposition_module.lstm_other.weight_ih))
    fr2 = c2.train.optim_config.freeze
    print('  phase 2 trainable modules', sorted({n.split('.')[0] for n, _ in ns2.named_parameters() if not any(f in n for f in fr2)}))
print('== 4. forward pass of the NS phase-2 model (eval runner, 4 generated pairs, A-2 hidden input)')
with h5py.File('data/data/v15_s2_genx_p80/hideA2_test.h5', 'r') as f:
    sv = f['train']['self_vision_in'][:4]
with h5py.File('data/data/v15_s2_genx_p80/data.h5', 'r') as f:
    sm = f['train']['self_motion'][:4]
ns2.eval()
out = E.run(ns2, sv, sm, torch.device('cpu'), pmask=lambda t: np.full(4, 0. if t == 0 else 1.))
print({k: (tuple(v.shape), bool(np.isfinite(v).all())) for k, v in out.items() if hasattr(v, 'shape')})
print('== 5. NS split module is used (h2 changes when lstm_other is perturbed, h1 does not)')
with torch.no_grad():
    for n, q in ns2.named_parameters():
        if 'lstm_other' in n: q.add_(0.05)
out2 = E.run(ns2, sv, sm, torch.device('cpu'), pmask=lambda t: np.full(4, 0. if t == 0 else 1.))
print('h1 max diff', float(np.abs(out['h1'] - out2['h1']).max()), 'h2 max diff', float(np.abs(out['h2'] - out2['h2']).max()))
