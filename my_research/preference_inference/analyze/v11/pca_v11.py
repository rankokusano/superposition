"""
v11 PCA state maps (Sec.7.3 of the updated instructions; docs/v11_experiment_log.md
Sec.7.18 item 18). For one seed and epoch, both conditions, on the common test
(v11_s1_random test, 300 ep, canonical masks, eval_seed 0): PCA of h1 (fit on all frames of both layouts) with h1 AND h2 projected on the h1 PCs
(the paper's plot_state.py map_by_self), plotted per layout and coloured by
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
# corner colours of the paper's analyze_util.make2Dcolormap (index [x, y]):
# (x max, y max) yellow, (x min, y max) blue, (x min, y min) green, (x max, y min) red;
# bilinear in between (the paper builds it with cv2.resize of the 2x2 corners).
CORNERS = {(1, 1): (1, 1, 0), (0, 1): (0, 0, 1), (0, 0): (0, 1, 0), (1, 0): (1, 0, 0)}


def color2d(pos):
    u = np.clip((pos[:, 0] + 10) / 20, 0, 1)[:, None]
    v = np.clip((pos[:, 1] + 10) / 20, 0, 1)[:, None]
    c = lambda k: np.array(CORNERS[k])[None]  # noqa
    return (1 - u) * (1 - v) * c((0, 0)) + u * (1 - v) * c((1, 0)) + (1 - u) * v * c((0, 1)) + u * v * c((1, 1))


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
        H1 = d['h'][:, ::STRIDE].reshape(-1, 128)
        H2 = d['h2'][:, ::STRIDE].reshape(-1, 128)
        mean1 = H1.mean(0)
        _, sg1, vt1 = np.linalg.svd(H1 - mean1, full_matrices=False)
        P1, P2on1 = (H1 - mean1) @ vt1[:2].T, (H2 - mean1) @ vt1[:2].T     # paper: PCA of h1, both layers projected
        mean2 = H2.mean(0)
        _, sg2, vt2 = np.linalg.svd(H2 - mean2, full_matrices=False)
        P2 = (H2 - mean2) @ vt2[:2].T                                        # supplementary: h2's own PCA
        s['pc12_var_h1'] = (sg1[:2] ** 2 / (sg1 ** 2).sum()).round(4).tolist()
        s['pc12_var_h2_own'] = (sg2[:2] ** 2 / (sg2 ** 2).sum()).round(4).tolist()
        s['plane_r2'] = {
            'h1_on_h1PC_A1pos_x': plane_r2(P1, sp[:, 0]), 'h1_on_h1PC_A1pos_y': plane_r2(P1, sp[:, 1]),
            'h2_on_h1PC_A2pos_x': plane_r2(P2on1, op[:, 0]), 'h2_on_h1PC_A2pos_y': plane_r2(P2on1, op[:, 1]),
            'h1_on_h1PC_reward': plane_r2(P1, reward), 'h1_on_h1PC_dist_red': plane_r2(P1, dist_red),
            'h2_ownPC_A2pos_x': plane_r2(P2, op[:, 0]), 'h2_ownPC_A2pos_y': plane_r2(P2, op[:, 1])}
        panels = [('h1 on h1-PCs, colour = A-1 position (2D)', P1, color2d(sp), None),
                  ('h2 on h1-PCs, colour = A-2 position (2D)', P2on1, color2d(op), None),
                  ('h1 on h1-PCs, colour = pleasure r_t', P1, reward, 'PiYG_r'),
                  ('h1 on h1-PCs, colour = dist A-1 to red', P1, dist_red, 'viridis'),
                  ('h2 own PCs (suppl.), colour = A-2 position (2D)', P2, color2d(op), None)]
        lim = np.r_[np.minimum(P1.min(0), P2on1.min(0)), np.maximum(P1.max(0), P2on1.max(0))]
        fig, axes = plt.subplots(2, len(panels), figsize=(4.4 * len(panels), 8.4))
        for k in (0, 1):
            sel = lay == k
            for j, (name, P, col, cmap) in enumerate(panels):
                a_ = axes[k, j]
                if cmap is None:
                    a_.scatter(P[sel, 0], P[sel, 1], c=col[sel], s=3, alpha=0.9, rasterized=True)
                else:
                    sc = a_.scatter(P[sel, 0], P[sel, 1], c=col[sel], s=3, cmap=cmap, alpha=0.7, rasterized=True)
                    plt.colorbar(sc, ax=a_)
                if j < 4:
                    a_.set_xlim(lim[0], lim[2]); a_.set_ylim(lim[1], lim[3])
                a_.set_title(f'{LAYNAME[k]}: {name}', fontsize=8)
                a_.set_xlabel('PC1'); a_.set_ylabel('PC2')
        fig.suptitle(f'v11 PCA state map (paper Fig.4c style: PCA of h1, h1 and h2 projected): '
                     f'{args.exp_fmt.format(cond)} s{args.seed} ep{args.epoch}, common random test, every {STRIDE}th step. '
                     f'2D colour corners: (+x,+y) yellow, (-x,+y) blue, (-x,-y) green, (+x,-y) red', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'pca_{cond}_s{args.seed}_ep{args.epoch}.png'), dpi=90)
        plt.close(fig)
        np.savez_compressed(os.path.join(out, f'pca_{cond}_s{args.seed}_ep{args.epoch}.npz'), P1=P1, P2on1=P2on1, P2=P2,
                            sp=sp, op=op, reward=reward, dist_red=dist_red, layout=lay)
        summary[cond] = s
        print(cond, json.dumps(s), flush=True)
    with open(os.path.join(out, f'pca_{args.exp_fmt.format("X")}_s{args.seed}_ep{args.epoch}.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
