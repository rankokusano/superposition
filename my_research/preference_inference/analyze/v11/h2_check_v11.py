"""
v11 diagnostic of h2 -> A-2 position (not pre-registered; user's request 2026-10-18,
docs/v11_experiment_log.md Sec.7.24). Common test (canonical masks, eval_seed 0).
Per (condition, seed, epoch): 5-fold GroupKFold (episodes) R^2 of A-2 x and y
separately from h2 with Ridge(1) (linear) and k-NN (k=10, non-linear), per layout
as well; explained variance of h2's top PCs. A-2 is static, so every frame of an
episode shares the target.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import DATA, LAYNAME, RES, load_model, run  # noqa

STRIDE = 5


def cv(model_fn, X, y, g):
    yh = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, g):
        yh[te] = model_fn().fit(X[tr], y[tr]).predict(X[te])
    return [float(1 - ((y[:, c] - yh[:, c]) ** 2).sum() / ((y[:, c] - y[:, c].mean()) ** 2).sum()) for c in (0, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='cond:seed:epoch')
    ap.add_argument('--sub', default='main/h2_check')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    path = os.path.join(DATA, 'v11_s1_random', 'data.h5')
    res = {}
    for r in args.runs:
        c, s, e = r.split(':')
        model, p_mask = load_model(f'v11_s1_{c}_mse', int(s), int(e), dev)
        d = run(model, p_mask, path, 'test', 0, dev, keep_h2=True)
        N = d['h2'].shape[0]
        X = d['h2'][:, ::STRIDE].reshape(-1, 128)
        y = d['other_pos'][:, :100:STRIDE].reshape(-1, 2)
        g = np.repeat(np.arange(N)[:, None], 100 // STRIDE, 1).reshape(-1)
        lay = np.repeat(d['layout'][:, None], 100 // STRIDE, 1).reshape(-1)
        Xc = X - X.mean(0)
        sg = np.linalg.svd(Xc, compute_uv=False)
        o = {'ridge_xy': cv(lambda: Ridge(alpha=1.0), X, y, g),
             'knn10_xy': cv(lambda: KNeighborsRegressor(10), X, y, g),
             'ridge_xy_per_layout': {LAYNAME[k]: cv(lambda: Ridge(alpha=1.0), X[lay == k], y[lay == k], g[lay == k]) for k in (0, 1)},
             'h2_pc_explained_var_top5': (sg[:5] ** 2 / (sg ** 2).sum()).round(4).tolist(),
             'h2_total_var': float((Xc ** 2).sum(1).mean())}
        res[r] = o
        print(r, json.dumps(o), flush=True)
    with open(os.path.join(out, 'h2_check.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
