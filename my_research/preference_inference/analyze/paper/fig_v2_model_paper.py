"""
Short-version figure 1 'v2_model' (paper round 3, work 2 and 5; documents only, CPU): environment (left) and network
(right) in one figure drawn at the text width of jsarticle 11pt A4 (406.86 pt = 5.65 in), so that the smallest text is
7 pt at print size. Content is the same as figs/env_conditions and figs/network (same sources; the env part shows the
view at (-8, 8) for the original layout only), with shorter box labels:
  (a) arena seen from above, original and red/green swapped; (b) A-1's view at (-8, 8) and P_k per bin;
  (c) A-1 occupancy in the stage-I training data (no experience / 50% / 80%, two layouts);
  (d) stage I (inset: no-sharing condition); (e) stage II (all but G frozen).
Writes paper/figs/v2_model.{pdf,png}.
"""

import os
import sys

import h5py
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from paper_style import PI, plt, save, setup

sys.path.insert(0, PI)
from sim_v10 import layout as L  # noqa

FS = 7.0                        # smallest text size (pt) at print size
TEXTWIDTH_IN = 406.86 / 72.27
DATA = {'none': 'v11_s1_random', '50': 'v11_s1_mixed', '80': 'v13_s1_mixed80'}
CLAB = {'none': '経験なし', '50': '経験50%', '80': '経験80%'}
COL = {'Red': '#e8302a', 'Green': '#2fb52f', 'Blue': '#2a4fd6', 'Cyan': '#22c8d8'}
PT = (-8, 8)
TRAIN, FROZEN, HATCH, EC = 'white', '#e6e6e6', '////', '0.25'
ORANGE, GREEN, BLUE = '#c45a1c', '#1baf7a', '#2a78d6'


# ------------------------------------------------------------------ network (compact)
def box(ax, x, y, w, h, text, frozen=False, fc=None, ec=EC, lw=0.7, z=2, color='k'):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.1',
                                fc=fc or (FROZEN if frozen else TRAIN), ec=ec, lw=lw, hatch=HATCH if frozen else None,
                                zorder=z))
    if text:
        ax.text(x, y, text, ha='center', va='center', fontsize=FS, zorder=z + 1, color=color, linespacing=1.05,
                bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.85) if frozen else None)


def arrow(ax, a, b, color='0.2', lw=0.7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=6, color=color, lw=lw, zorder=6,
                                 shrinkA=0, shrinkB=0))


def network(ax, stage):
    s2 = stage == 2
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 10)
    ax.axis('off')
    box(ax, 6, 0.55, 4.6, 0.75, 'A-1 の視界 $v_t$', fc='#fafafa')
    for x, n in ((4.3, 1), (7.7, 2)):
        box(ax, x, 1.9, 3.0, 0.75, f'視覚エンコーダ{n}', frozen=True)
        arrow(ax, (6 + (x - 6) * 0.45, 0.93), (x, 1.52))
        arrow(ax, (x, 2.28), (x, 3.12))
        ax.text(x + 0.12, 2.7, 'マスク', fontsize=FS, ha='left', va='center', color='0.35')
    box(ax, 6, 3.95, 7.4, 1.9, None, frozen=s2, z=1)
    ax.text(6, 4.6, '共有 LSTM（重みを共有）', ha='center', va='center', fontsize=FS, zorder=4,
            bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.85) if s2 else None)
    box(ax, 4.3, 3.65, 3.0, 1.0, 'process-1\n$h^1_t$', frozen=s2, z=3)
    box(ax, 7.7, 3.65, 3.0, 1.0, 'process-2\n$h^2_t$', frozen=s2, z=3)
    box(ax, 1.0, 3.65, 2.0, 2.0, '$x^1_t$\n運動 2\n快・不快 8\nフラグ 1', fc='#fff6ee', ec=ORANGE)
    arrow(ax, (2.0, 3.65), (2.8, 3.65), color=ORANGE)
    if s2:
        box(ax, 11.05, 3.65, 2.1, 1.0, '$x^2_t$\n$[0,Q2_t,1]$', fc='#fff6ee', ec=ORANGE)
        box(ax, 11.05, 1.9, 2.1, 1.5, '他者用 $G$\nLSTMCell\n→ 8', ec=GREEN, lw=1.2)
        arrow(ax, (9.2, 1.9), (10.0, 1.9), color=GREEN)
        ax.text(9.6, 1.05, 'マスク前', fontsize=FS, ha='center', va='center', color=GREEN)
        arrow(ax, (11.05, 2.65), (11.05, 3.15), color=GREEN)
    else:
        box(ax, 11.05, 3.65, 2.1, 1.0, '$x^2_t$\n零', fc='#fff6ee', ec=ORANGE)
    arrow(ax, (10.0, 3.65), (9.2, 3.65), color=ORANGE)
    box(ax, 6, 6.3, 5.4, 1.7, None, frozen=s2, z=1)
    ax.text(4.0, 6.85, '視覚予測器', ha='center', va='center', fontsize=FS, zorder=4,
            bbox=dict(fc='white', ec='none', pad=0.3, alpha=0.85) if s2 else None)
    box(ax, 4.6, 6.05, 2.4, 0.7, '統合層', frozen=s2, z=3)
    box(ax, 7.4, 6.05, 2.4, 0.7, '復号部', frozen=s2, z=3)
    arrow(ax, (5.8, 6.05), (6.2, 6.05))
    arrow(ax, (4.3, 4.15), (4.45, 5.7))
    arrow(ax, (7.7, 4.15), (5.0, 5.7))
    box(ax, 7.4, 8.15, 3.2, 0.7, '予測 $\\hat{v}_{t+1}$', fc='#fafafa')
    arrow(ax, (7.4, 6.4), (7.4, 7.8))
    box(ax, 11.05, 6.3, 2.1, 0.7, '特徴予測器', frozen=s2)
    arrow(ax, (9.7, 4.6), (10.3, 5.95))
    if not s2:
        box(ax, 1.95, 8.6, 3.7, 2.1, None, fc='#f7fbff', ec=BLUE, lw=0.8, z=1)
        ax.text(1.95, 9.3, '差し込み：非共有条件', fontsize=FS, ha='center', va='center', color=BLUE)
        box(ax, 1.15, 8.55, 1.5, 0.7, 'LSTM 1', ec=BLUE, z=3, color=BLUE)
        box(ax, 2.75, 8.55, 1.5, 0.7, 'LSTM 2', ec=BLUE, z=3, color=BLUE)
        ax.text(1.95, 7.85, '重みを共有しない', fontsize=FS, ha='center', va='center', color=BLUE)


# ------------------------------------------------------------------ environment (compact)
def topview(ax, lay):
    ax.add_patch(Rectangle((-10, -10), 20, 20, fill=False, lw=0.7, color='k'))
    for name, c in L.LANDMARKS[lay].items():
        ax.add_patch(Rectangle((c[0] - 2, c[1] - 2), 4, 4, color=COL[name], lw=0))
    ax.plot(*PT, marker='^', color='k', ms=4)
    ax.set_xlim(-15.5, 15.5)
    ax.set_ylim(-15.5, 15.5)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    setup()
    plt.rcParams.update({'font.size': FS, 'axes.titlesize': FS + 0.5, 'axes.labelsize': FS, 'xtick.labelsize': FS,
                         'ytick.labelsize': FS, 'legend.fontsize': FS})
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 6.2))
    outer = fig.add_gridspec(1, 2, width_ratios=(1.75, 3.85), wspace=0.04)
    gl = outer[0].subgridspec(3, 1, height_ratios=(1.3, 1.25, 2.4), hspace=0.55)
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
    # (c) rows = conditions, columns = layouts
    gc = gl[2].subgridspec(3, 3, width_ratios=(1, 1, 0.09), hspace=0.08, wspace=0.08)
    edges = np.linspace(-10, 10, 21)
    for row, cn in enumerate(('none', '50', '80')):
        g = h5py.File(os.path.join(PI, 'data', 'data', DATA[cn], 'data.h5'), 'r')['train']
        pos, lay = g['self_position'][()], g['layout'][()]
        for col, (ln, lname) in enumerate((('original', '元の配置'), ('swap_rg', '入れ替え'))):
            pp = pos[lay == col].reshape(-1, 2)
            H, _, _ = np.histogram2d(pp[:, 0], pp[:, 1], bins=[edges, edges])
            ax = fig.add_subplot(gc[row, col])
            im = ax.imshow((H / H.sum()).T, origin='lower', extent=[-10, 10, -10, 10], cmap='viridis', vmin=0, vmax=0.02)
            for name in ('Red', 'Green'):
                ax.plot(*np.clip(L.LANDMARKS[ln][name], -9.3, 9.3), 'o', color=COL[name], mec='w', ms=3.5, mew=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(lname, fontsize=FS)
            if col == 0:
                ax.set_ylabel(CLAB[cn], fontsize=FS, labelpad=2)
            if row == 0 and col == 0:
                ax.text(-0.25, 1.38, '(c) 学習データの滞在位置', transform=ax.transAxes, fontsize=FS + 0.5)
    cax = fig.add_subplot(gc[:, 2])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.01, 0.02])
    cb.ax.tick_params(labelsize=FS, length=2, pad=1)
    # (d)(e)
    gr = outer[1].subgridspec(3, 1, height_ratios=(1, 1, 0.1), hspace=0.1)
    for k, (stage, title) in enumerate(((1, '(d) 段階I：自己の予測学習'), (2, '(e) 段階II：他者用モジュール（$G$ 以外は凍結）'))):
        ax = fig.add_subplot(gr[k])
        network(ax, stage)
        ax.set_title(title, fontsize=FS + 0.5, loc='left', pad=1)
    axl = fig.add_subplot(gr[2])
    axl.axis('off')
    hs = [FancyBboxPatch((0, 0), 1, 1, fc=TRAIN, ec=EC), FancyBboxPatch((0, 0), 1, 1, fc=FROZEN, ec=EC, hatch=HATCH)]
    axl.legend(hs, ['その段階で学習', '凍結（L1 期間は復号部だけを学習）'], loc='center', ncol=2, frameon=False,
               fontsize=FS, handlelength=1.4, columnspacing=1.0)
    save(fig, 'v2_model')


if __name__ == '__main__':
    main()
