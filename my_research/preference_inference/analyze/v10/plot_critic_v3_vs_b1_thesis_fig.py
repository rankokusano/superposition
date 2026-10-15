"""
Thesis figure: the critic before (v3, trained in the original layout only;
stage A) and after training with the red/green swap (B1 seed0), for A-1
(likes red) and A-2 (likes green). Rows = critic, columns = Q-value map
(max over the 8 probe actions) in the original / swapped layout, and the
best-direction map (argmax over K=64) in the original / swapped layout.
21x21 grid, the other agent at (0, 0) (as plot_q_map_v6.py).
Sources: stageA/direction_maps/direction_maps.npz (v3) and
b1/<agent>_seed0/b1_arrays.npz (B1). Outputs PNG (300 dpi) + PDF to
data/result/v10_swap/b1/.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

COL = {'Red': '#d62728', 'Green': '#2ca02c', 'Blue': '#1f4fd6', 'Cyan': '#17becf'}
LM = {
    'original': {'Red': (-9, 9), 'Green': (-9, -9), 'Blue': (9, -9), 'Cyan': (9, 9)},
    'swap_rg': {'Red': (-9, -9), 'Green': (-9, 9), 'Blue': (9, -9), 'Cyan': (9, 9)},
}
LONG = {'original': 'original layout', 'swap_rg': 'red/green swapped'}


def landmarks(ax, lo):
    for nm, (x, y) in LM[lo].items():
        ax.scatter(x, y, marker='s', s=70, c=COL[nm], edgecolors='k', linewidths=0.6, zorder=5)
    ax.set_xlim(-10.8, 10.8)
    ax.set_ylim(-10.8, 10.8)
    ax.set_aspect('equal')
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])


def angdiff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def figure(agent, target, v3, b1, fname):
    grid = v3['grid']
    X, Y = np.meshgrid(grid, grid, indexing='ij')
    rows = [('v3 critic (trained in the original layout only)', lambda lo, k: v3[f'{agent.upper()}_{lo}_{k}']),
            ('B1 critic, seed 0 (trained with the swap)', lambda lo, k: b1[f'{lo}_{k}'])]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6), constrained_layout=True)
    for r, (rlab, get) in enumerate(rows):
        q = {lo: get(lo, 'qmax8') for lo in LM}
        lo_, hi_ = min(v.min() for v in q.values()), max(v.max() for v in q.values())
        q8o, q8s = get('original', 'q8').reshape(-1), get('swap_rg', 'q8').reshape(-1)
        corr = np.corrcoef(q8o, q8s)[0, 1]
        for c, lo in enumerate(LM):
            ax = axes[r, c]
            im = ax.imshow(q[lo].T, origin='lower', cmap='hot', extent=[-10.5, 10.5, -10.5, 10.5],
                           vmin=lo_, vmax=hi_)
            landmarks(ax, lo)
            ax.set_title(f'Q value, {LONG[lo]}', fontsize=10)
        fig.colorbar(im, ax=axes[r, 1], shrink=0.8, label='max Q over 8 probe actions')
        axes[r, 0].set_ylabel(f'{rlab}\n\ny', fontsize=10)
        axes[r, 0].text(-10.3, -12.9, f'corr. of Q(s, a) between layouts: {corr:+.3f}', fontsize=9, va='top')
        for c, lo in enumerate(LM):
            ax = axes[r, 2 + c]
            best = get(lo, 'best64')
            p = np.array(LM[lo][target])
            d = angdiff(best, np.degrees(np.arctan2(p[1] - Y, p[0] - X)))
            th = np.radians(best)
            qv = ax.quiver(X, Y, np.cos(th), np.sin(th), d, cmap='viridis', clim=(0, 180), scale=28, width=0.004)
            landmarks(ax, lo)
            ax.set_title(f'best action direction, {LONG[lo]}', fontsize=10)
        fig.colorbar(qv, ax=axes[r, 3], shrink=0.8, label=f'angle to {target.lower()} (deg)')
    fig.suptitle(f'{agent.upper().replace("A", "A-")} critic (likes {target.lower()}): '
                 'value follows the place before the swap training, and the colour after it '
                 '(other agent fixed at the centre)', fontsize=11)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_ROOT, 'b1', f'{fname}.{ext}'), dpi=300)
    plt.close(fig)


def main():
    v3 = np.load(os.path.join(OUT_ROOT, 'stageA', 'direction_maps', 'direction_maps.npz'))
    for agent, target in (('a1', 'Red'), ('a2', 'Green')):
        b1 = np.load(os.path.join(OUT_ROOT, 'b1', f'{agent}_seed0', 'b1_arrays.npz'))
        figure(agent, target, v3, b1, f'fig_critic_v3_vs_b1seed0_{agent}')
    print('saved')


if __name__ == '__main__':
    main()
