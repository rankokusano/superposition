"""
Paper v3 figure 1 'v3_model' (2026-11-01): figure 1 of the short version (fig_v2_model_paper.py, unchanged) without
the occupancy panel, which goes to the supplement as 's_occupancy'. Same drawing functions and sources:
  v3_model   : (a) arena seen from above, original and red/green swapped; (b) A-1's view at (-8, 8) and P_k per bin;
               (c) stage I (inset: no-sharing condition); (d) stage II (all but G frozen).
  s_occupancy: A-1 occupancy in the stage-I training data (no experience / 50% / 80%, two layouts) = old panel (c).
Drawn at the text width (406.86 pt) so that the smallest text is 7 pt at print size.
Writes paper/figs/v3_model.{pdf,png} and paper/figs/s_occupancy.{pdf,png}.
"""

import os

import h5py
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from fig_v2_model_paper import (CLAB, COL, DATA, EC, FROZEN, FS, HATCH, PT, TEXTWIDTH_IN, TRAIN, L, network,
                                topview)
from paper_style import PI, plt, save, setup


def env_panels(fig, gl):
    # (a)
    ga = gl[0].subgridspec(2, 2, height_ratios=(1, 0.42), hspace=0.05, wspace=0.08)
    for j, (lay, name) in enumerate((('original', '元の配置'), ('swap_rg', '入れ替え'))):
        ax = fig.add_subplot(ga[0, j])
        topview(ax, lay)
        ax.set_title(('(a) ' if j == 0 else '') + name, fontsize=FS, loc='left' if j == 0 else 'center')
    axl = fig.add_subplot(ga[1, :])
    axl.axis('off')
    hs = [Rectangle((0, 0), 1, 1, color=COL[n]) for n in ('Red', 'Green', 'Blue', 'Cyan')]
    hs.append(plt.Line2D([], [], marker='^', color='k', ls='', ms=4))
    axl.legend(hs, ['赤', '緑', '青', 'シアン', '(b) の位置'], loc='center', ncol=3, frameon=False, fontsize=FS,
               handlelength=1.0, columnspacing=0.6, handletextpad=0.3)
    # (b)
    z = np.load(os.path.join(PI, 'data', 'result', 'v11', 'gate61', 'gate61_pleasure_views.npz'))
    i = [tuple(p) for p in z['pts']].index(PT)
    gb = gl[1].subgridspec(2, 1, height_ratios=(1, 1.05), hspace=0.08)
    ax = fig.add_subplot(gb[0])
    ax.imshow(z['views_original'][i], aspect='auto', interpolation='nearest', extent=(0, 64, 16, 0))
    for k in range(1, 8):
        ax.axvline(8 * k, color='w', lw=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('(b) (−8, 8) での視界と $P_k$', fontsize=FS + 0.5, loc='left')
    ax2 = fig.add_subplot(gb[1])
    p = z['vals_original'][i]
    ax2.bar(np.arange(8) * 8 + 4, p, width=6.5, color=['#e8302a' if x > 0 else '#2fb52f' for x in p])
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlim(0, 64)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xticks(np.arange(8) * 8 + 4)
    ax2.set_xticklabels([str(k) for k in range(8)])
    ax2.set_yticks([-0.4, 0, 0.4])
    ax2.set_xlabel('範囲 $k$', labelpad=1)
    ax2.set_ylabel('$P_k$', labelpad=1)


def model_figure():
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 5.6))
    outer = fig.add_gridspec(1, 2, width_ratios=(1.75, 3.85), wspace=0.04)
    gl = outer[0].subgridspec(3, 1, height_ratios=(1.3, 1.25, 0.9), hspace=0.55)
    env_panels(fig, gl)
    gr = outer[1].subgridspec(3, 1, height_ratios=(1, 1, 0.1), hspace=0.1)
    for k, (stage, title) in enumerate(((1, '(c) 段階I：自己の予測学習'), (2, '(d) 段階II：他者用モジュール（$G$ 以外は凍結）'))):
        ax = fig.add_subplot(gr[k])
        network(ax, stage)
        ax.set_title(title, fontsize=FS + 0.5, loc='left', pad=1)
    axl = fig.add_subplot(gr[2])
    axl.axis('off')
    hs = [FancyBboxPatch((0, 0), 1, 1, fc=TRAIN, ec=EC), FancyBboxPatch((0, 0), 1, 1, fc=FROZEN, ec=EC, hatch=HATCH)]
    axl.legend(hs, ['その段階で学習', '凍結（L1 期間は復号部だけを学習）'], loc='center', ncol=2, frameon=False,
               fontsize=FS, handlelength=1.4, columnspacing=1.0)
    save(fig, 'v3_model')


def occupancy_figure():
    # the old panel (c), alone: rows = layouts, columns = conditions, at 0.6 of the text width
    fig = plt.figure(figsize=(TEXTWIDTH_IN * 0.62, 2.45))
    gc = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 0.07), hspace=0.08, wspace=0.08)
    edges = np.linspace(-10, 10, 21)
    for col, cn in enumerate(('none', '50', '80')):
        g = h5py.File(os.path.join(PI, 'data', 'data', DATA[cn], 'data.h5'), 'r')['train']
        pos, lay = g['self_position'][()], g['layout'][()]
        for row, (ln, lname) in enumerate((('original', '元の配置'), ('swap_rg', '入れ替え'))):
            pp = pos[lay == row].reshape(-1, 2)
            H, _, _ = np.histogram2d(pp[:, 0], pp[:, 1], bins=[edges, edges])
            ax = fig.add_subplot(gc[row, col])
            im = ax.imshow((H / H.sum()).T, origin='lower', extent=[-10, 10, -10, 10], cmap='viridis', vmin=0, vmax=0.02)
            for name in ('Red', 'Green'):
                ax.plot(*np.clip(L.LANDMARKS[ln][name], -9.3, 9.3), 'o', color=COL[name], mec='w', ms=3.5, mew=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(CLAB[cn], fontsize=FS)
            if col == 0:
                ax.set_ylabel(lname, fontsize=FS, labelpad=2)
    cax = fig.add_subplot(gc[:, 3])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.01, 0.02])
    cb.ax.tick_params(labelsize=FS, length=2, pad=1)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.03)
    save(fig, 's_occupancy')


def main():
    setup()
    plt.rcParams.update({'font.size': FS, 'axes.titlesize': FS + 0.5, 'axes.labelsize': FS, 'xtick.labelsize': FS,
                         'ytick.labelsize': FS, 'legend.fontsize': FS})
    model_figure()
    occupancy_figure()


if __name__ == '__main__':
    main()
