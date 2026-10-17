"""
v11 M5 and M6 (docs/v11_experiment_log.md Sec.5.4; secondary indices, Sec.7.13)
on the common test (v11_s1_random test, 300 ep, canonical masks, eval_seed 0).
Pairing as v4: h1 after input frame t <-> quantities at frame t (t = 0..99).

M5  Ridge(alpha=1), 5-fold GroupKFold by episode (cross-validated R^2 over
    all held-out frames), targets: r_t (A-1's reward of v_t: red - green
    fraction), G_t = sum_k 0.9^k r_{t+k} (to the episode end; frames t <= 80),
    cos / sin of the direction from p_t to the red reference.
    Cross-layout: fit on original-layout episodes, test on swap episodes,
    and vice versa.
M6  layout read-out (v10 B4 method): StandardScaler + L2 logistic regression
    (C=1.0), 5-fold split by episode (KFold shuffle, RandomState(0)), at
    t = 1, 5, 10, 20, 50, 100 meaning "after the t-th input frame" (h index t-1).
Writes JSON under data/result/v11/<sub>/.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import DATA, LAYNAME, RES, load_model, run  # noqa
from sim_v10 import layout as L  # noqa
from v11core.pleasure import color_mask  # noqa

T_READ = (1, 5, 10, 20, 50, 100)


def r2(y, yh):
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def cv_r2(X, y, groups):
    yh = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, groups):
        yh[te] = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
    return r2(y, yh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sub', default='main/m5m6')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    path = os.path.join(DATA, 'v11_s1_random', 'data.h5')
    g = h5py.File(path, 'r')['test']
    sv = g['self_vision'][:, :100]
    rew = color_mask(sv, 'Red').mean((-1, -2)) - color_mask(sv, 'Green').mean((-1, -2))  # (N, 100)
    N = rew.shape[0]
    G = np.zeros_like(rew)
    acc = np.zeros(N)
    for t in range(99, -1, -1):
        acc = rew[:, t] + 0.9 * acc
        G[:, t] = acc
    S = {}
    for cond in ('random', 'mixed'):
        model, p_mask = load_model(args.exp_fmt.format(cond), args.seed, args.epoch, dev)
        d = run(model, p_mask, path, 'test', 0, dev)
        h = d['h']                                  # (N, 100, 128), index t = after frame t
        pos = d['pos'][:, :100]
        lay = d['layout']
        red = np.stack([L.LANDMARKS[LAYNAME[int(l)]]['Red'] for l in lay])[:, None, :]
        u = red - pos
        ang = np.arctan2(u[..., 1], u[..., 0])
        targets = {'reward_r_t': rew, 'return_G_t_gamma0.9': G, 'cos_red_dir': np.cos(ang), 'sin_red_dir': np.sin(ang)}
        groups = np.repeat(np.arange(N)[:, None], 100, 1)
        tmask = np.ones((N, 100), bool)
        s = {'M5_cv_r2': {}, 'M5_cross_layout_r2': {}}
        for name, y in targets.items():
            m = tmask.copy()
            if name.startswith('return'):
                m[:, 81:] = False
            X, yy, gg = h[m], y[m], groups[m]
            s['M5_cv_r2'][name] = cv_r2(X, yy, gg)
            lf = np.repeat(lay[:, None], 100, 1)[m]
            cr = {}
            for a, b in ((0, 1), (1, 0)):
                reg = Ridge(alpha=1.0).fit(X[lf == a], yy[lf == a])
                cr[f'fit_{LAYNAME[a]}_test_{LAYNAME[b]}'] = r2(yy[lf == b], reg.predict(X[lf == b]))
            s['M5_cross_layout_r2'][name] = cr
        acc6 = {}
        for t in T_READ:
            X = h[:, t - 1]
            pred = np.zeros(N)
            for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
                pred[te] = clf.fit(X[tr], lay[tr]).predict(X[te])
            acc6[f't{t}'] = float((pred == lay).mean())
        s['M6_layout_readout_acc'] = acc6
        S[cond] = s
        print(cond, json.dumps(s), flush=True)
    with open(os.path.join(out, f'm5m6_{args.exp_fmt.format("X")}_s{args.seed}_ep{args.epoch}.json'), 'w') as f:
        json.dump(S, f, indent=1)


if __name__ == '__main__':
    main()
