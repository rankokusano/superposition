"""
v11 PCA state maps (Sec.7.3 of the updated instructions; docs/v11_experiment_log.md
Sec.7.18 item 18). For one seed and epoch, both conditions, on the common test
(v11_s1_random test, 300 ep, canonical masks, eval_seed 0): PCA (fit on all
frames of both layouts) of h1 (process-1) and h2 (process-2), plotted per
layout and coloured by
  pos-x, pos-y (as plot_pca_state_v4.py), 2D position colour (x -> red,
  y -> green), pleasure r_t (A-1's reward: red - green fraction of v_t),
  distance from A-1 to the red reference (h1 only; h2 rows use A-2's position).
Also the linear R^2 of the PC1-2 plane for each colour variable.
Every 5th time step (as v4). Writes PNG per condition and a JSON under
data/result/v11/<sub>/.
"""

import argparse
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import DATA, LAYNAME, RES, load_model, run  # noqa
from sim_v10 import layout as L  # noqa
from v11core.pleasure import color_mask  # noqa

STRIDE = 5


def pca2(x, mean, vt):
    return (x - mean) @ vt[:2].T


def plane_r2(P, y):
    X = np.c_[P, np.ones(len(P))]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - (y - X @ beta).var() / y.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sub', default='pca')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    path = os.path.join(DATA, 'v11_s1_random', 'data.h5')
    g = h5py.File(path, 'r')['test']
    sv = g['self_vision'][:, :100:STRIDE]
    red = color_mask(sv, 'Red').mean((-1, -2))
    grn = color_mask(sv, 'Green').mean((-1, -2))
    reward = (red - grn).reshape(-1)
    summary = {}
    for cond in ('random', 'mixed'):
        model, p_mask = load_model(args.exp_fmt.format(cond), args.seed, args.epoch, dev)
        d = run(model, p_mask, path, 'test', 0, dev, keep_h2=True)
        lay = np.repeat(d['layout'][:, None], 100 // STRIDE, 1).reshape(-1)
        sp = d['pos'][:, :100:STRIDE].reshape(-1, 2)
        op = d['other_pos'][:, :100:STRIDE].reshape(-1, 2)
        refs = np.stack([L.LANDMARKS[LAYNAME[int(l)]]['Red'] for l in lay])
        dist_red = np.linalg.norm(sp - refs, axis=1)
        s = {}
        fig, axes = plt.subplots(4, 5, figsize=(22, 16))
        for li, (hk, pos, who) in enumerate((('h', sp, 'h1'), ('h2', op, 'h2'))):
            H = d[hk][:, ::STRIDE].reshape(-1, 128)
            mean = H.mean(0)
            _, sing, vt = np.linalg.svd(H - mean, full_matrices=False)
            ev = sing[:2] ** 2 / (sing ** 2).sum()
            P = pca2(H, mean, vt)
            cols = [('pos-x', pos[:, 0], 'coolwarm'), ('pos-y', pos[:, 1], 'coolwarm'), ('2D pos', None, None),
                    ('pleasure r_t', reward, 'PiYG_r'), ('dist A-1 to red', dist_red, 'viridis')]
            s[who] = {'pc12_var': ev.round(4).tolist(),
                      'plane_r2': {name: plane_r2(P, v) for name, v, _ in cols if v is not None}}
            for k in (0, 1):
                sel = lay == k
                row = 2 * li + k
                for j, (name, v, cmap) in enumerate(cols):
                    a = axes[row, j]
                    if v is None:
                        c2 = np.c_[(pos[sel, 0] + 10) / 20, (pos[sel, 1] + 10) / 20, 0.5 * np.ones(sel.sum())]
                        a.scatter(P[sel, 0], P[sel, 1], c=np.clip(c2, 0, 1), s=3, rasterized=True)
                    else:
                        sc = a.scatter(P[sel, 0], P[sel, 1], c=v[sel], s=3, cmap=cmap, alpha=0.6, rasterized=True)
                        plt.colorbar(sc, ax=a)
                    a.set_title(f'{who} / {LAYNAME[k]}: colour = {name}' + (' (A-2 pos)' if li and 'pos' in name else ''),
                                fontsize=9)
                    a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f'v11 PCA state map: {args.exp_fmt.format(cond)} s{args.seed} ep{args.epoch}, common random test '
                     f'(PCA fit on both layouts; every {STRIDE}th step). 2D pos: x -> red, y -> green', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'pca_{cond}_s{args.seed}_ep{args.epoch}.png'), dpi=90)
        plt.close(fig)
        summary[cond] = s
        print(cond, json.dumps(s), flush=True)
    with open(os.path.join(out, f'pca_{args.exp_fmt.format("X")}_s{args.seed}_ep{args.epoch}.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
