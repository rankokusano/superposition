"""
v11 data figure (instructions Sec.7.2 "top view and real images"; gate 6.1 item 6).
For the same viz20 episode number in both conditions (paired: same layout,
start position, first random target and A-2 position), per layout:
  left: top view of the arena (landmark cubes at their true centres +-13 in the
        project colours, reference points (+-9, +-9), A-1's trajectory with each
        step coloured by a1_source: rule = red, random = grey, rule->random
        fallback = orange, asleep = black dot; A-2's static position);
  right: A-1's real view and A-2's real view at t = 1, 5, 10, 20, 50.
Real images are the stored arrays shown as RGB (v10 convention; project Red/Green
look red/green, sky brown). Writes PNGs under data/result/v11/<sub>/.
"""

import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from sim_v10 import layout as L  # noqa

DATA = os.path.join(_PI_DIR, 'data', 'data')
T_IMG = (1, 5, 10, 20, 50)
COL = {'Red': 'red', 'Green': 'limegreen', 'Blue': 'blue', 'Cyan': 'cyan'}
SRC_COL = {0: '0.55', 1: 'red', 2: 'orange'}


def top_view(ax, pos, src, op, layout_name, title):
    ax.add_patch(plt.Rectangle((-10, -10), 20, 20, fill=False, lw=1.5))
    for name, c in L.CUBE_CENTERS[layout_name].items():
        ax.add_patch(plt.Rectangle(c - 3, 6, 6, color=COL[name], alpha=0.8))
        r = L.LANDMARKS[layout_name][name]
        ax.plot(*r, 'x', color='k', ms=5)
    for t in range(len(pos) - 1):
        if src[t] == 3:
            ax.plot(*pos[t], '.', color='k', ms=3)
        else:
            ax.plot(pos[t:t + 2, 0], pos[t:t + 2, 1], '-', color=SRC_COL[int(src[t])], lw=1.5)
    ax.plot(*pos[0], 'o', mfc='white', mec='k', ms=8)
    ax.plot(*op[0], 's', color='k', ms=9)
    ax.set_xlim(-17, 17); ax.set_ylim(-17, 17); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sub', default='data_figs')
    ap.add_argument('--suffix', default='viz20')
    args = ap.parse_args()
    out = os.path.join(_PI_DIR, 'data', 'result', 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    G = {c: h5py.File(os.path.join(DATA, f'v11_s1_{c}_{args.suffix}', 'data.h5'), 'r')['train'] for c in ('random', 'mixed')}
    lay = G['random']['layout'][()]
    assert np.array_equal(lay, G['mixed']['layout'][()])
    for code in (0, 1):
        ep = int(np.where(lay == code)[0][0])
        name = L.CODE_LAYOUT[code]
        fig = plt.figure(figsize=(22, 8.5))
        gs = fig.add_gridspec(4, 1 + len(T_IMG), width_ratios=[2.2] + [1.6] * len(T_IMG))
        for i, c in enumerate(('random', 'mixed')):
            g = G[c]
            pos, src, op = g['self_position'][ep], g['a1_source'][ep], g['other_position'][ep]
            ax = fig.add_subplot(gs[2 * i:2 * i + 2, 0])
            frac = (src[:100] == 1).sum() / max(1, (src[:100] != 3).sum())
            top_view(ax, pos, src, op, name, f'{c} / {name} / viz20 episode {ep}\nA-1 start o, A-2 square; rule steps {frac:.2f}')
            for j, t in enumerate(T_IMG):
                for r, (key, lab) in enumerate((('self_vision', 'A-1 view'), ('other_vision', 'A-2 view'))):
                    a_ = fig.add_subplot(gs[2 * i + r, 1 + j])
                    a_.imshow(g[key][ep, t], aspect='auto', interpolation='nearest')
                    a_.set_xticks([]); a_.set_yticks([])
                    a_.set_title(f'{c}: {lab}, t={t}, A-1 at ({pos[t, 0]:.1f},{pos[t, 1]:.1f})', fontsize=7)
        fig.suptitle('v11 data: top view (trajectory colour = a1_source: red rule, grey random, orange rule->random fallback, '
                     'black dots asleep; x = reference points (+-9,+-9)) and real views (array channels shown as RGB, v10 convention)',
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'data_topview_{name}_ep{ep}.png'), dpi=100)
        plt.close(fig)
        print('saved', name, ep)


if __name__ == '__main__':
    main()
