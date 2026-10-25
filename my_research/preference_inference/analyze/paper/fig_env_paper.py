"""
Paper figure 2 'env_conditions' (documents only, CPU).
(a) schematic top view: arena [-10, 10]^2, landmark cubes centred at (+-13, +-13) (sim_v10/layout.py; the cube size
    in the drawing is schematic), original layout and red/green swapped; the A-1 position of (b) is marked.
(b) A-1's panoramic view (16 x 64, array shown as RGB as in data/result/v11/gate61/gate61_pleasure_views.png) at
    (-8, 8) in both layouts, with the 8 bins (columns 8k..8k+7) and the pleasure P_k = (liked red - disliked green)/128,
    from data/result/v11/gate61/gate61_pleasure_views.npz (analyze/v11/gate61_v11.py).
(c) A-1 occupancy in the stage-I training data (train 3000 episodes, all frames, fraction per 1 x 1 cell), no experience /
    50% / 80%, per layout - same computation as analyze/v13/gate61_v13.py (gate61_v13_occupancy.png).
Writes paper/figs/env_conditions.{pdf,png}.
"""

import os
import sys

import h5py
import numpy as np
from matplotlib.patches import Rectangle

from paper_style import PI, pair_label, plt, save, setup  # noqa

sys.path.insert(0, PI)
from sim_v10 import layout as L  # noqa

DATA = {'none': 'v11_s1_random', '50': 'v11_s1_mixed', '80': 'v13_s1_mixed80'}
CLAB = {'none': '経験なし', '50': '経験50%', '80': '経験80%'}
LAY = (('original', '元の配置'), ('swap_rg', '赤と緑を入れ替えた配置'))
COL = {'Red': '#e8302a', 'Green': '#2fb52f', 'Blue': '#2a4fd6', 'Cyan': '#22c8d8'}
PT = (-8, 8)
CUBE = 4.0


def topview(ax, lay, mark):
    ax.add_patch(Rectangle((-10, -10), 20, 20, fill=False, lw=0.8, color='k'))
    for name, c in L.LANDMARKS[lay].items():
        ax.add_patch(Rectangle((c[0] - CUBE / 2, c[1] - CUBE / 2), CUBE, CUBE, color=COL[name], lw=0))
    ax.plot(*mark, marker='^', color='k', ms=5)
    ax.set_xlim(-15.5, 15.5)
    ax.set_ylim(-15.5, 15.5)
    ax.set_aspect('equal')
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    ax.tick_params(labelsize=6.5)


def main():
    setup()
    z = np.load(os.path.join(PI, 'data', 'result', 'v11', 'gate61', 'gate61_pleasure_views.npz'))
    i = [tuple(p) for p in z['pts']].index(PT)
    fig = plt.figure(figsize=(6.3, 7.3))
    outer = fig.add_gridspec(3, 1, height_ratios=(1.0, 1.25, 1.65), hspace=0.55)
    # (a)
    ga = outer[0].subgridspec(1, 2, wspace=0.05)
    for j, (lay, lname) in enumerate(LAY):
        ax = fig.add_subplot(ga[0, j])
        topview(ax, lay, PT)
        ax.set_title(f'({"ab"[j]}) ' + lname + '（上から見た図）', fontsize=8.5)
        if j == 1:
            hs = [Rectangle((0, 0), 1, 1, color=COL[n]) for n in ('Red', 'Green', 'Blue', 'Cyan')]
            hs.append(plt.Line2D([], [], marker='^', color='k', ls='', ms=5))
            ax.legend(hs, ['赤', '緑', '青', 'シアン', '(c)・(d) の A-1 の位置'], loc='center left',
                      bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    # (b)
    gb = outer[1].subgridspec(2, 2, height_ratios=(1, 1.1), hspace=0.08, wspace=0.18)
    for j, (lay, lname) in enumerate(LAY):
        v = z['views_original' if j == 0 else 'views_swap'][i]
        p = z['vals_original' if j == 0 else 'vals_swap'][i]
        ax = fig.add_subplot(gb[0, j])
        ax.imshow(v, aspect='auto', interpolation='nearest', extent=(0, 64, 16, 0))
        for k in range(1, 8):
            ax.axvline(8 * k, color='w', lw=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'({"cd"[j]}) ' + f'{lname}\nA-1 の位置 ({PT[0]}, {PT[1]}) の視界と快・不快', fontsize=8)
        ax2 = fig.add_subplot(gb[1, j])
        ax2.bar(np.arange(8) * 8 + 4, p, width=6.5, color=['#e8302a' if x > 0 else '#2fb52f' for x in p])
        ax2.axhline(0, color='k', lw=0.5)
        ax2.set_xlim(0, 64)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xticks(np.arange(8) * 8 + 4)
        ax2.set_xticklabels([str(k) for k in range(8)])
        ax2.set_xlabel('範囲 $k$', fontsize=7.5, labelpad=1)
        if j == 0:
            ax2.set_ylabel('快・不快 $P_k$')
    # (c)
    gc = outer[2].subgridspec(2, 3, hspace=0.12, wspace=-0.45)
    edges = np.linspace(-10, 10, 21)
    axs_c = []
    for col, cn in enumerate(('none', '50', '80')):
        g = h5py.File(os.path.join(PI, 'data', 'data', DATA[cn], 'data.h5'), 'r')['train']
        pos, lay = g['self_position'][()], g['layout'][()]
        for row, (ln, lname) in enumerate(LAY):
            pp = pos[lay == row].reshape(-1, 2)
            H, _, _ = np.histogram2d(pp[:, 0], pp[:, 1], bins=[edges, edges])
            ax = fig.add_subplot(gc[row, col])
            im = ax.imshow((H / H.sum()).T, origin='lower', extent=[-10, 10, -10, 10], cmap='viridis', vmin=0, vmax=0.02)
            for name in ('Red', 'Green'):
                c = np.clip(L.LANDMARKS[ln][name], -9.3, 9.3)
                ax.plot(*c, 'o', color=COL[name], mec='w', ms=5, mew=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            axs_c.append(ax)
            if row == 0:
                ax.set_title(CLAB[cn], fontsize=8.5)
            if col == 0:
                ax.set_ylabel('元の配置' if row == 0 else '入れ替えた配置', fontsize=7.5)
    cb = fig.colorbar(im, ax=axs_c, fraction=0.03, pad=0.01)
    b = axs_c[0].get_position()
    fig.text(b.x0 - 0.06, b.y1 + 0.03, '(e) 段階Iの学習データにおける A-1 の滞在位置', fontsize=9)
    cb.set_label('フレームの割合（1×1 の区画）', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, 'env_conditions')


if __name__ == '__main__':
    main()
