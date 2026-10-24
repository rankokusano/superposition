"""
T3's D6 split by A-2's step (docs/v13_experiment_log.md Sec.24; outside the pre-registration, no training).
Stage-1 models (ep200 of the l1 stage = the non-G weights of the stage-2 models): random = v12_s1_random_l1_s<s>,
mixed50 = v12_s1_mixed_l1_s<s>, mixed80 = v13_s1_mixed_l1_s<s>. Protocol of analyze/v11/t1t3_s2_v11.run:
util.seed_all(0), vision at t = 0 only, process-1 motion shown (no hiding), process-2 = [0, 0, P_A2(t), 1]
('d6') or [0, 0, 0, 1] ('zero'); per frame e_t = MSE(predict_feature(h2_t), Encoder-2(x_{t+1})).
Writes data/result/v13/post/d6_split/s<s>/<cond>.npz with e (N, 100) per set and variant.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _PI)
from eval_trial_v12 import RES, load_v12  # noqa
from eval_s2_v11 import load_set, true_pleasure  # noqa
import util  # noqa

EXPS = {'random': 'v12_s1_random_l1_s{s}', 'mixed50': 'v12_s1_mixed_l1_s{s}', 'mixed80': 'v13_s1_mixed_l1_s{s}'}


@torch.no_grad()
def run(model, d, P, dev, batch=50):
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    E = []
    util.seed_all(0)
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b0 + batch]).float().to(dev)
        B = x.shape[0]
        model.init_state(B)
        e = []
        for t in range(100):
            pm = 0 if t == 0 else 1
            q = torch.zeros(B, 11, device=dev)
            q[:, 10] = 1.0
            if P is not None:
                q[:, 2:10] = torch.from_numpy(P[b0:b0 + batch, t]).float().to(dev)
            model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=q)
            os_ = model.get_state()['other'][0]
            tgt = model.other_vision_encoder_module(x[:, t + 1])
            e.append(F.mse_loss(model.predict_feature(os_), tgt, reduction='none').mean(1).cpu().numpy())
        E.append(np.stack(e, 1))
    return np.concatenate(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    args = ap.parse_args()
    out = os.path.join(RES, 'v13', 'post', 'd6_split', f's{args.seed}')
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in ('green', 'pref')}
    PA2 = {k: true_pleasure(k, D[k]) for k in D}
    for c, exp in EXPS.items():
        m, _ = load_v12(exp.format(s=args.seed), args.seed, 200, dev)
        m.motion_masking = False     # stage-2 protocol: process-1 motion is not hidden
        r = {}
        for k in D:
            r[f'{k}_d6'] = run(m, D[k], PA2[k], dev)
            r[f'{k}_zero'] = run(m, D[k], None, dev)
            r[f'{k}_a2_source'] = D[k]['a2_source'][:, :100]
        np.savez_compressed(os.path.join(out, f'{c}.npz'), **r)
        print(c, {q: round(float(v.mean()), 4) for q, v in r.items() if not q.endswith('source')}, flush=True)


if __name__ == '__main__':
    main()
