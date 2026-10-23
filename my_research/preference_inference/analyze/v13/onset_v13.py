"""
v13 onset vs total movement (docs/v13_experiment_log.md Sec.12.1; descriptive, outside the
pre-registration, gates unchanged). Uses the gate functions unchanged:
  G3 : eval_trial_v12.freeze (K = 15, pleasure held) + localise_v11.dstar -> d*_X(k), X = R, G, B, C;
       onset = d*_R(1) - mean_others d*(1), total = d*_R(15) - mean_others d*(15).
  G5a: eval_g5_v12.g5a_est ('freeze', K = 15, main read-out) -> P_X(k) = (est_k - p2(t0)) . u_X;
       onset = P_liked(1) - mean_others P(1), total = P_liked(15) - mean_others P(15); sets green / pref.
Models: no experience v12_s1_random_*, experienced 50% v12_s1_mixed_*, experienced 80% v13_s1_mixed_*
(G3: mse stage, G5a: l1 stage of the same seed). Writes per-episode values
data/result/v13/main/onset/s<s>/onset_<cond>_ep<e>.npz (cond random / mixed50 / mixed80).
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
import eval_trial_v12 as etv  # noqa
from eval_g5_v12 import COLORS, T0, calibrate, g5a_est, load_set, refs, true_pleasure  # noqa
from localise_v11 import dstar  # noqa

RES = etv.RES
EXPS = {'random': ('v12_s1_random_mse', 'v12_s1_random_l1_s{s}'),
        'mixed50': ('v12_s1_mixed_mse', 'v12_s1_mixed_l1_s{s}'),
        'mixed80': ('v13_s1_mixed_mse', 'v13_s1_mixed_l1_s{s}')}
K = 15


def proj_liked_minus_others(est, d):
    N = est.shape[0]
    p0 = d['other_position'][:, T0]
    liked = np.where(d['a2_liked'] == 1, 'Red', 'Green')
    li = np.array([COLORS.index(l) for l in liked])
    P = []
    for X in COLORS:
        u = refs(d['layout'], X)[:, 0] - p0
        u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
        P.append(((est - p0[:, None]) * u[:, None]).sum(-1))    # (N, K)
    P = np.stack(P, 1)                                           # (N, 4, K)
    pl = P[np.arange(N), li]
    po = np.stack([np.delete(P[n], li[n], axis=0).mean(0) for n in range(N)])
    return pl - po                                               # (N, K); NaN where unread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--conds', nargs='+', default=list(EXPS))
    args = ap.parse_args()
    out = os.path.join(RES, 'v13', 'main', 'onset', f's{args.seed}')
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    fv = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_dist_views.h5'), 'r')
    views, vvalid = fv['views'][()], fv['valid'][()]
    D = {k: load_set(k) for k in ('green', 'pref')}
    PA2 = {k: true_pleasure(k, D[k]) for k in D}
    g = h5py.File(D['green']['path'], 'r')['test']
    alpha, beta = calibrate(g['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    for e in args.epochs:
        for c in args.conds:
            r = {}
            m, pm = etv.load_v12(EXPS[c][0], args.seed, e, dev)
            ds = dstar(etv.freeze(m, pm, dev, K, True), views, vvalid)   # (N, 4, K)
            diff = ds[:, 0] - np.nanmean(ds[:, 1:], 1)
            r['G3_onset'], r['G3_total'] = diff[:, 0], diff[:, K - 1]
            m, pm = etv.load_v12(EXPS[c][1].format(s=args.seed), args.seed, e, dev)
            m.motion_masking = False     # stage-2 protocol, as in the G5a gate
            for k in D:
                est = g5a_est(m, D[k], PA2[k], dev, alpha, beta, K, 'freeze')
                pd = proj_liked_minus_others(est, D[k])
                r[f'G5a_{k}_onset'], r[f'G5a_{k}_total'] = pd[:, 0], pd[:, K - 1]
            np.savez_compressed(os.path.join(out, f'onset_{c}_ep{e}.npz'), **r)
            print(c, e, {q: round(float(np.nanmean(v)), 3) for q, v in r.items()}, flush=True)


if __name__ == '__main__':
    main()
