"""
Paper round 24 Sec.1.3, 1.4, 1.6 (paper/check/paper_instrctions_round24.md; docs/paper_extra_log.md Sec.48;
exploratory; NO training: only short timing trials whose weights and values are neither used nor saved). GPU (GPU1-3).
New code only: the existing model / runner / trainer files are imported, not changed.

R24Model = model_v12.SuperpositionNetworkMotionPleasureHoldOther (the stage-II model: stage-I predictor + G) with an
adapter on the other pathway's pleasure input: p2 = [0, 0, A(P2), 1], where P2 is G's output (or the override's
pleasure part) and A is
  'linear'  8 -> 8 nn.Linear, weight = identity, bias = 0 (exactly the identity at the start);
  'res16'   P2 + W2 relu(W1 P2), W1 8 -> 16 (He init), W2 16 -> 8 initialised to 0 (exactly the identity at the start).
Modes:
  inventory  weights of the stage-II model by role (self only / other only / shared), names, shapes, counts;
  identity   R24Model (both adapters, identity start) vs the original model, stage-II flow (vision masked for t > 0,
             MotionGenerationPredictionRunner), v11_s2_pref test episodes 0-49, eval mode: max |diff| of the
             predicted view and of q2_hat at every step (also train mode with the same seed);
  grad       stage-1 form (trainable: adapter + G) and stage-2 form (+ superposition LSTM): one stage-II loss on a
             batch of 10 training episodes, backward, one AdamW step; which parameters get a gradient and which change;
  loss       the batch keys the runner feeds (x, truth) and the loss terms;
  time       short trials (default 20 iterations of batch 10 x 100 steps) of the stage-1 / stage-2 forms: seconds per
             training iteration and per no-grad evaluation iteration, peak GPU memory.
-> data/result/paper_extra/Round24/model/<mode>.json
Usage: python analyze/paper_extra/round24_model.py <mode> [--form stage1|stage2] [--iters 20] [--tag x]
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
os.chdir(_PI)
import util  # noqa
from model import model_v12  # noqa
from model.model_v12 import MOTION_DIM  # noqa
from model import util as mutil  # noqa  (mask, as model_v12)
from exp import runner as runners, loader  # noqa

OUT = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round24', 'model')
STAGE2 = 'v12_s2_pref_mixed_s0'
DATA = os.path.join(_PI, 'data', 'data', 'v11_s2_pref', 'data.h5')


class R24Adapter(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == 'linear':
            self.fc = nn.Linear(8, 8)
            with torch.no_grad():
                self.fc.weight.copy_(torch.eye(8))
                self.fc.bias.zero_()
        else:
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 8)
            mutil.init_fc(self.fc1, 'relu', 'he')
            with torch.no_grad():
                self.fc2.weight.zero_()
                self.fc2.bias.zero_()

    def forward(self, q):
        if self.kind == 'linear':
            return self.fc(q)
        return q + self.fc2(F.relu(self.fc1(q)))


class R24Model(model_v12.SuperpositionNetworkMotionPleasureHoldOther):
    def __init__(self, config, kind='linear'):
        super().__init__(config)
        self.p2_adapter = R24Adapter(kind)

    def forward(self, x, p_mask_vision_self, p_mask_vision_other, p1_override=None, p2_override=None,
                hidden_override=None):
        sv = x['self_vision']
        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)
        p1 = self.process1_input(x, p1_override, hidden_override)
        q2 = self.motion_generator_module(ov_enc)
        if p2_override is None:
            p2 = torch.cat([torch.zeros_like(q2[:, :MOTION_DIM]), self.p2_adapter(q2), torch.ones_like(q2[:, :1])], 1)
        else:
            p2 = torch.cat([p2_override[:, :2], self.p2_adapter(p2_override[:, 2:10]), p2_override[:, 10:]], 1)
        self._t += 1
        sv_enc = mutil.mask(sv_enc, p_mask_vision_self)
        ov_enc = mutil.mask(ov_enc, p_mask_vision_other)
        ss, os_ = self.superposition_module(sv_enc, p1, ov_enc, p2)
        so = self.integration_module(F.dropout(ss, p=0.5, training=self.training),
                                     F.dropout(os_, p=0.5, training=self.training))
        return {'self_vision': self.vision_decoder_module(so), 'q2_hat': q2}


def configs():
    exp_cfg = util.load_config(os.path.join(_PI, 'config', 'exp', STAGE2 + '.yml'))
    mcfg = util.load_config(os.path.join(_PI, 'config', 'model', exp_cfg.model.name, exp_cfg.model.config + '.yml'))
    return exp_cfg, mcfg


def stage2_state():
    return torch.load(os.path.join(_PI, 'data', 'result', STAGE2, '0', 'model', '00200.pth'), map_location='cpu')['model']


def build(kind, dev):
    exp_cfg, mcfg = configs()
    if kind is None:
        m = model_v12.SuperpositionNetworkMotionPleasureHoldOther(mcfg)
        m.load_state_dict(stage2_state())
    else:
        m = R24Model(mcfg, kind)
        res = m.load_state_dict(stage2_state(), strict=False)
        assert not res.unexpected_keys and all(k.startswith('p2_adapter') for k in res.missing_keys), res
    return m.to(dev), exp_cfg


def data(split, n0, n1, dev):
    g = h5py.File(DATA, 'r')[split]
    d = {k: g[k][n0:n1] for k in ('self_vision', 'self_motion', 'self_position', 'other_motion', 'other_position', 'other_vision')}
    return loader.DataLoader(d, 10, False, dev), {k: g[k][n0:n1] for k in ('a2_liked',)}


def role(name):
    if name.startswith('self_vision_encoder_module'):
        return 'self_only'
    if name.startswith(('other_vision_encoder_module', 'motion_generator_module', 'p2_adapter')):
        return 'other_only'
    return 'shared'


def inventory(dev):
    m, _ = build(None, dev)
    rows = [{'name': n, 'shape': list(p.shape), 'numel': p.numel(), 'role': role(n)} for n, p in m.named_parameters()]
    tot = {}
    for r in rows:
        tot[r['role']] = tot.get(r['role'], 0) + r['numel']
    mods = {}
    for r in rows:
        k = r['name'].split('.')[0]
        mods.setdefault(k, {'role': r['role'], 'n_tensors': 0, 'numel': 0})
        mods[k]['n_tensors'] += 1
        mods[k]['numel'] += r['numel']
    return {'model': STAGE2 + ' ep200 (SuperpositionNetworkMotionPleasureHoldOther)', 'totals': tot, 'modules': mods,
            'params': rows,
            'buffers': [{'name': n, 'shape': list(b.shape)} for n, b in m.named_buffers()]}


@torch.no_grad()
def run_flow(m, dl, seed=0):
    util.seed_all(seed)
    outs, qs = [], []
    for seq, _ in dl.load_sequence():
        for x, truth, t in seq:
            if t == 0:
                m.init_state(x['self_vision'].shape[0])
            p = 0 if t == 0 else 1
            o = m(x, p, p)
            outs.append(o['self_vision'].cpu())
            qs.append(o['q2_hat'].cpu())
    return torch.stack(outs), torch.stack(qs)


def identity(dev):
    dl, _ = data('test', 0, 50, dev)
    ref, _ = build(None, dev)
    out = {}
    for mode in ('eval', 'train'):
        getattr(ref, mode)()
        a, qa = run_flow(ref, dl)
        for kind in ('linear', 'res16'):
            m, _ = build(kind, dev)
            getattr(m, mode)()
            b, qb = run_flow(m, dl)
            out[f'{mode}|{kind}'] = {'max_abs_diff_self_vision': float((a - b).abs().max()),
                                     'bit_identical_self_vision': bool(torch.equal(a, b)),
                                     'max_abs_diff_q2_hat': float((qa - qb).abs().max()),
                                     'bit_identical_q2_hat': bool(torch.equal(qa, qb))}
    out['tf32_matmul'] = bool(torch.backends.cuda.matmul.allow_tf32)
    out['episodes'] = 'v11_s2_pref test 0-49, 100 steps, vision masked for t > 0 (stage-II flow)'
    return out


def trainable(name, form):
    keep = ('p2_adapter', 'motion_generator_module') + (('superposition_module',) if form == 'stage2' else ())
    return name.startswith(keep)


def setup_train(kind, form, dev):
    m, exp_cfg = build(kind, dev)
    for n, p in m.named_parameters():
        p.requires_grad = trainable(n, form)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
    return m, exp_cfg, opt


def loss_seq(m, seq, exp_cfg, runner):
    loss = {}
    from collections import defaultdict
    loss = defaultdict(lambda: 0)
    for x, truth, t in seq:
        if t == 0:
            m.init_state(x['self_vision'].shape[0])
        p = runner.gen_mask_prob(t, exp_cfg)
        pred = m(x, p, p)
        runner.calc_loss(m, loss, pred, truth, 0, t, exp_cfg)
    return loss


def grad(dev):
    dl, _ = data('train', 0, 10, dev)
    runner = runners.MotionGenerationPredictionFeaturePredictionRunner()
    out = {}
    for form in ('stage1', 'stage2'):
        for kind in ('linear', 'res16'):
            util.seed_all(0)
            m, exp_cfg, opt = setup_train(kind, form, dev)
            m.train()
            before = {n: p.detach().clone() for n, p in m.named_parameters()}
            seq, _ = next(iter(dl.load_sequence()))
            loss = loss_seq(m, seq, exp_cfg, runner)
            opt.zero_grad()
            sum(loss.values()).backward()
            g = {}
            for n, p in m.named_parameters():
                k = n.split('.')[0]
                d = g.setdefault(k, {'trainable': trainable(n, form), 'grad_nonzero': 0, 'grad_none_or_zero': 0,
                                     'changed_after_step': 0, 'unchanged_after_step': 0})
                if p.grad is not None and p.grad.abs().sum() > 0:
                    d['grad_nonzero'] += 1
                else:
                    d['grad_none_or_zero'] += 1
            opt.step()
            for n, p in m.named_parameters():
                d = g[n.split('.')[0]]
                if torch.equal(before[n], p.detach()):
                    d['unchanged_after_step'] += 1
                else:
                    d['changed_after_step'] += 1
            ok = all((v['trainable'] and v['unchanged_after_step'] == 0) or
                     (not v['trainable'] and v['changed_after_step'] == 0 and v['grad_nonzero'] == 0) for v in g.values())
            out[f'{form}|{kind}'] = {'modules': g, 'only_intended_changed': ok,
                                     'loss_terms': {k: float(v) for k, v in loss.items()}}
            del m, opt
            torch.cuda.empty_cache()
    return out


def loss_keys(dev):
    dl, extra = data('train', 0, 10, dev)
    seq, _ = next(iter(dl.load_sequence()))
    x, truth, t = next(iter(seq))
    g = h5py.File(DATA, 'r')['train']
    import inspect
    runner = runners.MotionGenerationPredictionFeaturePredictionRunner
    return {'dataset_keys': sorted(g.keys()), 'x_keys': sorted(x.keys()), 'truth_keys': sorted(truth.keys()),
            'a2_liked_in_batch': 'a2_liked' in x or 'a2_liked' in truth,
            'model_forward_reads': ['self_vision', 'self_motion'],
            'calc_loss_source': inspect.getsource(runner.calc_loss),
            'base_calc_loss_source': inspect.getsource(runners.PredictionRunnerBase.calc_loss)}


def timing(dev, form, iters):
    n = 10 * (iters + 2)
    dl, _ = data('train', 0, n, dev)
    runner = runners.MotionGenerationPredictionFeaturePredictionRunner()
    m, exp_cfg, opt = setup_train('linear', form, dev)
    m.train()
    torch.cuda.reset_peak_memory_stats(dev)
    ts = []
    for i, (seq, _) in enumerate(dl.load_sequence()):
        torch.cuda.synchronize(dev)
        t0 = time.time()
        loss = loss_seq(m, seq, exp_cfg, runner)
        opt.zero_grad()
        sum(loss.values()).backward()
        opt.step()
        torch.cuda.synchronize(dev)
        ts.append(time.time() - t0)
    te = []
    m.eval()
    with torch.no_grad():
        for i, (seq, _) in enumerate(dl.load_sequence()):
            torch.cuda.synchronize(dev)
            t0 = time.time()
            loss_seq(m, seq, exp_cfg, runner)
            torch.cuda.synchronize(dev)
            te.append(time.time() - t0)
    return {'form': form, 'iters': len(ts), 'sec_per_train_iter_median': float(np.median(ts[2:])),
            'sec_per_eval_iter_median': float(np.median(te[2:])), 'peak_mem_MB': torch.cuda.max_memory_allocated(dev) / 2 ** 20,
            'batch': 10, 'steps': 100, 'n_trainable': int(sum(p.numel() for p in m.parameters() if p.requires_grad))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('inventory', 'identity', 'grad', 'loss', 'time'))
    ap.add_argument('--form', default='stage1')
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    os.makedirs(OUT, exist_ok=True)
    res = {'inventory': lambda: inventory(dev), 'identity': lambda: identity(dev), 'grad': lambda: grad(dev),
           'loss': lambda: loss_keys(dev), 'time': lambda: timing(dev, a.form, a.iters)}[a.mode]()
    name = a.mode + (f'_{a.form}' if a.mode == 'time' else '') + (f'_{a.tag}' if a.tag else '')
    json.dump(res, open(os.path.join(OUT, name + '.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('params', 'calc_loss_source', 'base_calc_loss_source')}
                     if isinstance(res, dict) else res, indent=1)[:6000])
