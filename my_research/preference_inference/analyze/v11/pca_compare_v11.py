"""
v11 PCA comparison across conditions and seeds (not pre-registered; user's request
2026-10-18, docs/v11_experiment_log.md Sec.7.24). Common test (v11_s1_random test,
canonical masks, eval_seed 0), every 5th step, PCA of h1 (fit on both layouts).
  - per PC (top 10): explained variance and Pearson r with A-1 x, y, pleasure r_t,
    distance to red (pooled and per layout);
  - linear R^2 of A-1 position / r_t / distance to red from the top-k PCs (k = 1..10);
  - figure: rows = seeds, columns = (random, mixed) x colour by 2D position,
    r_t, distance to red (PC1-PC2 plane, both layouts pooled, layout as marker).
Writes JSON and PNG under data/result/v11/<sub>/.
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
from pca_v11 import color2d  # noqa
from sim_v10 import layout as L  # noqa
from v11core.pleasure import color_mask  # noqa

STRIDE, NPC = 5, 10


def lin_r2(X, y):
    X1 = np.c_[X, np.ones(len(X))]
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    return float(1 - (res ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', default='v11_s1_{}_mse')
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--sub', default='main/pca_compare')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    path = os.path.join(DATA, 'v11_s1_random', 'data.h5')
    g = h5py.File(path, 'r')['test']
    sv = g['self_vision'][:, :100:STRIDE]
    reward = (color_mask(sv, 'Red').mean((-1, -2)) - color_mask(sv, 'Green').mean((-1, -2))).reshape(-1)
    res = {}
    fig, axes = plt.subplots(len(args.seeds), 6, figsize=(26, 4.2 * len(args.seeds)), squeeze=False)
    for i, s in enumerate(args.seeds):
        for j, cond in enumerate(('random', 'mixed')):
            model, p_mask = load_model(args.exp_fmt.format(cond), s, args.epoch, dev)
            d = run(model, p_mask, path, 'test', 0, dev)
            H = d['h'][:, ::STRIDE].reshape(-1, 128)
            lay = np.repeat(d['layout'][:, None], 100 // STRIDE, 1).reshape(-1)
            sp = d['pos'][:, :100:STRIDE].reshape(-1, 2)
            red = np.stack([L.LANDMARKS[LAYNAME[int(l)]]['Red'] for l in lay])
            dist = np.linalg.norm(sp - red, axis=1)
            Hc = H - H.mean(0)
            _, sg, vt = np.linalg.svd(Hc, full_matrices=False)
            P = Hc @ vt[:NPC].T
            ev = sg ** 2 / (sg ** 2).sum()
            vars_ = {'pos_x': sp[:, 0], 'pos_y': sp[:, 1], 'reward_r_t': reward, 'dist_red': dist}
            per_pc = []
            for k in range(NPC):
                row = {'explained_var': float(ev[k])}
                for name, v in vars_.items():
                    row[f'r_{name}'] = float(np.corrcoef(P[:, k], v)[0, 1])
                    for c in (0, 1):
                        m = lay == c
                        row[f'r_{name}_{LAYNAME[c]}'] = float(np.corrcoef(P[m, k], v[m])[0, 1])
                per_pc.append(row)
            topk = {name: [lin_r2(P[:, :k], (sp if name == 'position' else v)) for k in range(1, NPC + 1)]
                    for name, v in (('position', None), ('reward_r_t', reward), ('dist_red', dist))}
            topk['position_all128'] = lin_r2(H, sp)
            res[f'{cond}_s{s}'] = {'per_pc': per_pc, 'topk_linear_r2': topk}
            for c, (name, col, cmap) in enumerate((('A-1 position (2D)', color2d(sp), None),
                                                   ('pleasure r_t', reward, 'PiYG_r'),
                                                   ('dist A-1 to red', dist, 'viridis'))):
                a_ = axes[i, 3 * j + c]
                for lc, mk in ((0, 'o'), (1, '^')):
                    m = lay == lc
                    kw = dict(s=3, alpha=0.6, marker=mk, rasterized=True)
                    if cmap is None:
                        a_.scatter(P[m, 0], P[m, 1], c=col[m], **kw)
                    else:
                        sc = a_.scatter(P[m, 0], P[m, 1], c=col[m], cmap=cmap, vmin=col.min(), vmax=col.max(), **kw)
                if cmap is not None:
                    plt.colorbar(sc, ax=a_)
                pc = per_pc
                a_.set_title(f'{cond} s{s}: {name}\nPC1 r(x,y,rt,d)={pc[0]["r_pos_x"]:+.2f},{pc[0]["r_pos_y"]:+.2f},'
                             f'{pc[0]["r_reward_r_t"]:+.2f},{pc[0]["r_dist_red"]:+.2f}  '
                             f'PC2={pc[1]["r_pos_x"]:+.2f},{pc[1]["r_pos_y"]:+.2f},{pc[1]["r_reward_r_t"]:+.2f},{pc[1]["r_dist_red"]:+.2f}',
                             fontsize=7)
                a_.set_xlabel(f'PC1 ({ev[0]:.2f})'); a_.set_ylabel(f'PC2 ({ev[1]:.2f})')
            print(cond, s, 'ev', np.round(ev[:5], 3), 'top2 R2 pos/rt/dist', [round(topk[k][1], 3) for k in ('position', 'reward_r_t', 'dist_red')],
                  'all128 pos', round(topk['position_all128'], 3), flush=True)
    fig.suptitle('v11 PCA of h1 (paper style axes), mse ep200, common random test; circles = original, triangles = swap_rg. '
                 'Titles: Pearson r of PC1/PC2 with (A-1 x, A-1 y, r_t, dist to red)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'pca_compare_ep{args.epoch}.png'), dpi=90)
    with open(os.path.join(out, f'pca_compare_ep{args.epoch}.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
