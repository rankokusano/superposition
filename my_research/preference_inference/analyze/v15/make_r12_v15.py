"""
v15 Sec.38 (pre-registered before any GPU run): start checkpoints for the last check "reset only the other side's 11 input
columns" (no-experience, seeds 0-2; phase 2 = the 2(b) design, G only). Written as phase-1-like checkpoints
data/result/v15_s2p1_noexp_v15i_<tag>/<seed>/model/00150.pth, read by util.load_pretrain like a real phase 1:
    cp : NS form, every weight = the shared noexp phase 1 (ep150), other-side LSTM = an exact copy of the shared LSTM
         (check only, not trained)
    r1 : cp with lstm_other.weight_ih[:, 64:] (the 11 motion / pleasure / flag columns) = the same-seed NS noexp phase-1
         lstm_other.weight_ih[:, 64:] (initial values after weight decay only; zero gradient in phase 1, log Sec.33.8)
    r2 : the same-seed NS noexp phase 1 (ep150) with lstm_other.weight_ih[:, 64:] = the shared lstm.weight_ih[:, 64:]
Checks (CPU, stop on failure):
    1. r1 / r2: every tensor except the 11 replaced columns equals its source exactly; the 11 columns equal the donor.
    2. Loaded as in train_v12.py (util.seed_all(seed), build the NS model, util.load_pretrain with the phase-2 config), the
       state equals the checkpoint; G (not in a phase-1 checkpoint, random init from the seed) equals the shared phase-2 ep0.
    3. cp loaded the same way gives the same ep0 outputs as the shared model's phase-2 ep0 on the 930 A-2-hidden pairs
       (h1, h2, q2, G's state, predicted-view error; max abs diff <= 1e-6, bit-identical reported).

    python analyze/v15/make_r12_v15.py   # -> data/result/v15/stage2/make_r12_v15.json
"""

import copy
import json
import os
import sys

import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
import util  # noqa: E402
from model import model_v15, model_v15ns  # noqa: E402
for _name in ('SuperpositionNetworkMotionPleasureHoldV15NS', 'SuperpositionNetworkMotionPleasureHoldOtherV15NS'):
    setattr(model_v15, _name, getattr(model_v15ns, _name))
from analyze.v15 import eval_s2_hide_v15 as H  # noqa: E402
from analyze.v15 import eval_s2_v15 as E  # noqa: E402

RES = os.path.join(_PI_DIR, 'data', 'result')
NV = 64
P = 'superposition_module.'
WI = P + 'lstm_other.weight_ih'
TOL = 1e-6


def ck(exp, seed, epoch):
    return torch.load(os.path.join(RES, exp, str(seed), 'model', f'{epoch:05d}.pth'), map_location='cpu')['model']


def suf(s):
    return '' if s == 0 else f'_s{s}'


def build(s):
    sh = ck('v15_s2p1_noexp_v15i_mse', s, 150)
    ns = ck('v15_s2p1_noexp_v15i_ns_mse', s, 150)
    assert set(ns) == set(sh) | {k.replace(P + 'lstm.', P + 'lstm_other.') for k in sh if k.startswith(P + 'lstm.')}
    cp = copy.deepcopy(sh)
    for k in sh:
        if k.startswith(P + 'lstm.'):
            cp[k.replace(P + 'lstm.', P + 'lstm_other.')] = sh[k].clone()
    cp = {k: cp[k] for k in ns}  # NS key order
    r1 = copy.deepcopy(cp); r1[WI][:, NV:] = ns[WI][:, NV:]
    r2 = copy.deepcopy(ns); r2[WI][:, NV:] = sh[P + 'lstm.weight_ih'][:, NV:]
    return sh, ns, {'cp': cp, 'r1': r1, 'r2': r2}


def check_replaced(new, src, donor_cols):
    """-> (all other entries bit-equal to src, replaced 11 columns bit-equal to the donor, n tensors compared)"""
    other = all(torch.equal(new[k], src[k]) for k in src if k != WI) and torch.equal(new[WI][:, :NV], src[WI][:, :NV])
    return bool(other), bool(torch.equal(new[WI][:, NV:], donor_cols)), len(src)


def save(tag, s, sd):
    d = os.path.join(RES, f'v15_s2p1_noexp_v15i_{tag}', str(s), 'model')
    os.makedirs(d, exist_ok=True)
    torch.save({'model': sd}, os.path.join(d, '00150.pth'))


def load_like_training(p2exp, pre_tag, seed):
    util.seed_all(seed)  # as train_v12.py: seed, build, load_pretrain (no random draws in between)
    cfg = util.load_config(os.path.join(_PI_DIR, 'config', 'exp', p2exp + '.yml'))
    cfg.train.pretrain.exp_config = f'v15_s2p1_noexp_v15i_{pre_tag}'
    mcfg = util.load_config(os.path.join(_PI_DIR, 'config', 'model', cfg.model.name, cfg.model.config + '.yml'))
    m = getattr(model_v15, cfg.model.name)(mcfg)
    util.load_pretrain(m, cfg)
    return m.eval()


def main():
    torch.set_num_threads(2)
    dev = torch.device('cpu')
    both, te, pair, dropped = H.hidden_pairs('v15_s2_genx_p80', 'v15_s2_geny_p80')
    N = len(te)
    pm = lambda t: np.full(N, 0. if t == 0 else 1.)  # noqa: E731
    R = {'n_pairs': N // 2, 'dropped': dropped, 'tol': TOL}
    ok = True
    for s in (0, 1, 2):
        sh, ns, built = build(s)
        for tag, sd in built.items():
            save(tag, s, sd)
        r = {}
        r['r1_rest_equal_to_cp'], r['r1_cols_equal_ns'], _ = check_replaced(built['r1'], built['cp'], ns[WI][:, NV:])
        r['r2_rest_equal_to_ns'], r['r2_cols_equal_shared'], _ = check_replaced(built['r2'], ns, sh[P + 'lstm.weight_ih'][:, NV:])
        r['cp_self_lstm_equal_shared'] = all(torch.equal(built['cp'][k], sh[k]) for k in sh)
        r['rel_diff_11cols_r1_vs_shared'] = float((built['r1'][WI][:, NV:] - sh[P + 'lstm.weight_ih'][:, NV:]).norm()
                                                  / sh[P + 'lstm.weight_ih'][:, NV:].norm())
        r['norm_11cols'] = {'shared': float(sh[P + 'lstm.weight_ih'][:, NV:].norm()), 'ns_other': float(ns[WI][:, NV:].norm())}
        ref = E.load(f'v15_s2p2h_noexp_v15i{suf(s)}', 0, dev, seed=s)
        rs = ref.state_dict()
        r['shared_p2ep0_equals_p1ep150'] = all(torch.equal(rs[k], sh[k]) for k in sh)
        models = {}
        for tag in ('cp', 'r1', 'r2'):
            p2 = f'v15_s2p2h_noexp_v15i_{"r1" if tag == "cp" else tag}{suf(s)}'
            m = load_like_training(p2, tag, s)
            st = m.state_dict()
            rest = [k for k in st if k not in built[tag]]
            r[f'{tag}_loaded_equals_ckpt'] = all(torch.equal(st[k], built[tag][k]) for k in built[tag])
            r[f'{tag}_rest_is_G_equal_shared_ep0'] = all(k.startswith('motion_generator_module.') for k in rest) and \
                all(torch.equal(st[k], rs[k]) for k in rest)
            r[f'{tag}_n_G_tensors'] = len(rest)
            models[tag] = m
        a = E.run(ref, both['self_vision_in'], both['self_motion'], dev, pmask=pm, need_err=True)
        b = E.run(models['cp'], both['self_vision_in'], both['self_motion'], dev, pmask=pm, need_err=True)
        r['cp_vs_shared_ep0'] = {k: {'max_abs_diff': float(np.abs(a[k] - b[k]).max()), 'bit_identical': bool(np.array_equal(a[k], b[k]))}
                                 for k in ('h1', 'h2', 'q2', 'g', 'err') if k in a}
        passed = all(v for k, v in r.items() if isinstance(v, bool)) and \
            all(v['max_abs_diff'] <= TOL for v in r['cp_vs_shared_ep0'].values())
        r['passed'] = passed
        ok &= passed
        R[s] = r
        print(s, json.dumps(r), flush=True)
    R['all_passed'] = ok
    json.dump(R, open(os.path.join(E.OUT, 'make_r12_v15.json'), 'w'), indent=1)
    print('ALL_PASSED' if ok else 'FAILED', flush=True)


if __name__ == '__main__':
    main()
