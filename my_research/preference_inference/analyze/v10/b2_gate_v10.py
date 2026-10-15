"""
v10 gate B2 (docs/v10_experiment_log.md Sec.5.4; report only):
  - episodes per layout (train / test)
  - A-2's terminal position (t=100) per layout: distance to green / to the
    top-left (-9,9) / bottom-left (-9,-9), share within 5 of green
  - share of A-2 frames within 5 of green, per layout
  - A-1: correlation between "distance to red" and "distance to the top-left
    (-9,9)" over all train frames (pre-registered), plus per layout
Figures: A-2 terminal positions per layout; distance-to-red vs
distance-to-top-left scatter for A-1.
"""

import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR, git_hash  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

DATA = os.path.join(PI_DIR, 'data', 'data', 'v10_swapmix_r2', 'data.h5')
OUT = os.path.join(OUT_ROOT, 'b2')
TL, BL = np.array([-9.0, 9.0]), np.array([-9.0, -9.0])
NAME = {0: 'original', 1: 'swap_rg'}
RED = {0: TL, 1: BL}
GREEN = {0: BL, 1: TL}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {'dataset': 'v10_swapmix_r2', 'git_commit': git_hash(), 'splits': {}}
    with h5py.File(DATA, 'r') as f:
        res['attrs'] = {k: (v.item() if hasattr(v, 'item') else str(v)) for k, v in f.attrs.items()}
        d = {sp: {k: f[f'{sp}/{k}'][()] for k in ('layout', 'self_position', 'other_position')}
             for sp in ('train', 'test')}
    for sp, g in d.items():
        r = {'n_episodes': int(len(g['layout']))}
        for c in (0, 1):
            m = g['layout'] == c
            op = g['other_position'][m]
            term = op[:, -1]
            dg = np.linalg.norm(term - GREEN[c], axis=1)
            r[NAME[c]] = {
                'n_episodes': int(m.sum()),
                'A2_terminal_dist_to_green': {'mean': float(dg.mean()), 'median': float(np.median(dg))},
                'A2_terminal_dist_to_top_left': float(np.linalg.norm(term - TL, axis=1).mean()),
                'A2_terminal_dist_to_bottom_left': float(np.linalg.norm(term - BL, axis=1).mean()),
                'A2_terminal_within5_green': float((dg <= 5).mean()),
                'A2_frames_within5_green': float((np.linalg.norm(op - GREEN[c], axis=2) <= 5).mean()),
                'A2_terminal_mean_xy': term.mean(0).tolist(),
            }
        sp1 = g['self_position']
        red = np.stack([RED[c] for c in g['layout']])[:, None, :]
        d_red = np.linalg.norm(sp1 - red, axis=2)
        d_tl = np.linalg.norm(sp1 - TL, axis=2)
        r['A1_corr_dist_red_vs_dist_top_left'] = {
            'all_frames': float(np.corrcoef(d_red.ravel(), d_tl.ravel())[0, 1]),
            **{NAME[c]: float(np.corrcoef(d_red[g['layout'] == c].ravel(), d_tl[g['layout'] == c].ravel())[0, 1])
               for c in (0, 1)}}
        res['splits'][sp] = r
        if sp == 'train':
            plot(g, d_red, d_tl)
    json.dump(res, open(os.path.join(OUT, 'b2_gate.json'), 'w'), indent=2)
    print(json.dumps(res['splits'], indent=1))


def plot(g, d_red, d_tl):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, c in zip(axes[:2], (0, 1)):
        term = g['other_position'][g['layout'] == c][:, -1]
        h = ax.hist2d(term[:, 0], term[:, 1], bins=np.linspace(-10, 10, 21), cmap='Blues')
        fig.colorbar(h[3], ax=ax, shrink=0.8, label='episodes')
        ax.scatter(*RED[c], marker='s', s=90, c='#d62728', edgecolors='k')
        ax.scatter(*GREEN[c], marker='s', s=90, c='#2ca02c', edgecolors='k')
        ax.set_aspect('equal')
        ax.set_title(f'A-2 (B1 seed0 actor) terminal position, {NAME[c]} '
                     f'({int((g["layout"] == c).sum())} train ep)', fontsize=10)
    ax = axes[2]
    rng = np.random.RandomState(0)
    idx = rng.choice(d_red.size, 20000, replace=False)
    lay = np.repeat(g['layout'], d_red.shape[1])[idx]
    for c, col in ((0, '#4C78A8'), (1, '#F58518')):
        m = lay == c
        ax.scatter(d_tl.ravel()[idx][m], d_red.ravel()[idx][m], s=2, alpha=0.3, c=col, label=NAME[c])
    r = np.corrcoef(d_red.ravel(), d_tl.ravel())[0, 1]
    ax.set_xlabel('A-1 distance to top-left (-9, 9)')
    ax.set_ylabel('A-1 distance to red')
    ax.set_title(f'A-1 (all train frames; 20000 shown): r = {r:+.3f}', fontsize=10)
    ax.legend(markerscale=6, fontsize=8)
    ax.set_aspect('equal')
    fig.suptitle('v10 gate B2: v10_swapmix_r2')
    fig.savefig(os.path.join(OUT, 'b2_gate.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
