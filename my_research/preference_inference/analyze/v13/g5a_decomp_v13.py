"""
Post-hoc decomposition of the G5a slope (docs/v13_experiment_log.md Sec.15; outside the pre-registration,
no training, gates unchanged).
  (b) flipped: the G5a protocol of eval_g5_v12.g5a_est, but for t >= t0 process-2 carries -P_A2(t0);
      g_other by eval_g5_v12.g5a_metric (directions use the TRUE liked colour).
  (c) pre-freeze placement: the predicted frame t0 (made at step t0 - 1 from the true P_A2 history),
      A-2 read with read_a2; (est - p2) . u_liked and the same minus the mean over the 3 other colours.
      Reference: the same read-out on the TRUE view at t0.
  --check: also runs the 'freeze' variant with g5a_est and compares with the saved gate values.
Models: random = v12_s1_random_l1_s<s>, mixed50 = v12_s1_mixed_l1_s<s>, mixed80 = v13_s1_mixed_l1_s<s>.
Writes data/result/v13/post/g5a_decomp/s<s>/decomp_<cond>_ep<e>.npz (+ ref_true_view.npz once).
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _PI)
from eval_trial_v12 import RES, load_v12  # noqa
from eval_g5_v12 import COLORS, T0, calibrate, g5a_est, g5a_metric, load_set, read_a2, refs, true_pleasure  # noqa
import util  # noqa

EXPS = {'random': 'v12_s1_random_l1_s{s}', 'mixed50': 'v12_s1_mixed_l1_s{s}', 'mixed80': 'v13_s1_mixed_l1_s{s}'}
K = 15


@torch.no_grad()
def run_flip(model, d, PA2, dev, alpha, beta, batch=50):
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    util.seed_all(0)
    est, pre = [], []
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b0 + batch]).float().to(dev)
        sp = d['self_position'][b0:b0 + batch]
        B = x.shape[0]
        model.init_state(B)
        e = []
        for t in range(T0 + K):
            q = torch.zeros(B, 11, device=dev)
            q[:, 10] = 1.0
            q[:, 2:10] = torch.from_numpy(PA2[b0:b0 + batch, t] if t < T0 else -PA2[b0:b0 + batch, T0]).to(dev)
            pm = 0 if t == 0 else 1
            pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=q)['self_vision']
            if t == T0 - 1 or t >= T0:
                p01 = ((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                r = read_a2(p01, sp[:, t + 1], alpha, beta)
                if t == T0 - 1:
                    pre.append(r)       # predicted frame t0
                else:
                    e.append(r)
        est.append(np.stack(e, 1))
    return np.concatenate(est), np.concatenate(pre)


def placement(est0, d):
    """(est0 - p2) projected on u_liked, and liked minus the mean of the 3 other colours."""
    p0 = d['other_position'][:, T0]
    liked = np.where(d['a2_liked'] == 1, 'Red', 'Green')
    li = np.array([COLORS.index(l) for l in liked])
    P = []
    for X in COLORS:
        u = refs(d['layout'], X)[:, 0] - p0
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        P.append(((est0 - p0) * u).sum(-1))
    P = np.stack(P, 1)
    N = len(li)
    pl = P[np.arange(N), li]
    return pl, pl - np.stack([np.delete(P[n], li[n]).mean() for n in range(N)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--conds', nargs='+', default=list(EXPS))
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    out = os.path.join(RES, 'v13', 'post', 'g5a_decomp', f's{args.seed}')
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in ('green', 'pref')}
    PA2 = {k: true_pleasure(k, D[k]) for k in D}
    g = h5py.File(D['green']['path'], 'r')['test']
    alpha, beta = calibrate(g['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    ref_f = os.path.join(RES, 'v13', 'post', 'g5a_decomp', 'ref_true_view.npz')
    if not os.path.exists(ref_f):
        r = {}
        for k, d in D.items():
            v = h5py.File(d['path'], 'r')[d['split']]['self_vision'][:, T0].astype(np.float32) / 255.0
            r[f'{k}_liked'], r[f'{k}_liked_minus_others'] = placement(read_a2(v, d['self_position'][:, T0], alpha, beta), D[k])
        np.savez_compressed(ref_f, **r)
        print('ref', {q: round(float(np.nanmean(v)), 3) for q, v in r.items()}, flush=True)
    for e in args.epochs:
        for c in args.conds:
            m, _ = load_v12(EXPS[c].format(s=args.seed), args.seed, e, dev)
            m.motion_masking = False     # stage-2 protocol, as in the G5a gate
            r = {}
            for k in D:
                est, pre = run_flip(m, D[k], PA2[k], dev, alpha, beta)
                r[f'{k}_flip_g'] = g5a_metric(est, D[k])[0]
                r[f'{k}_pre_liked'], r[f'{k}_pre_liked_minus_others'] = placement(pre, D[k])
                if args.check:
                    gf = g5a_metric(g5a_est(m, D[k], PA2[k], dev, alpha, beta, K, 'freeze'), D[k])[0]
                    r[f'{k}_freeze_g_check'] = gf
            np.savez_compressed(os.path.join(out, f'decomp_{c}_ep{e}.npz'), **r)
            print(c, e, {q: round(float(np.nanmean(v)), 3) for q, v in r.items()}, flush=True)


if __name__ == '__main__':
    main()
