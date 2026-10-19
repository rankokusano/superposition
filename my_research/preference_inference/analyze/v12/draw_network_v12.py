"""
Network diagrams for v11 and v12 (instructions Sec.5.4), in the style of the paper's Fig.3/Fig.6:
boxes for the two encoders, the shared LSTM (process-1 / process-2), integration module, decoder and the
other's module G; the inputs of each process are written on the arrows. Frozen parts are hatched.
Writes data/result/v12/figs/network_v11_v12.png.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
from matplotlib.patches import FancyBboxPatch  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v12 import RES  # noqa


def box(ax, x, y, w, h, text, frozen=False, color='#dbe9f6'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02', fc=color, ec='k', lw=1,
                                hatch='///' if frozen else None))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7)


def arrow(ax, x0, y0, x1, y1, text='', off=(0, 0.02)):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=1))
    if text:
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], text, fontsize=6, ha='center', color='tab:blue')


def panel(ax, title, p1_text, p2_text, enc_frozen, stage2, other_frozen):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.set_title(title, fontsize=9)
    box(ax, 0.02, 0.45, 0.14, 0.1, "A-1's view v_t", color='#f4f4f4')
    box(ax, 0.24, 0.66, 0.16, 0.1, 'Encoder-1', frozen=enc_frozen)
    box(ax, 0.24, 0.24, 0.16, 0.1, 'Encoder-2', frozen=enc_frozen)
    arrow(ax, 0.16, 0.5, 0.24, 0.71); arrow(ax, 0.16, 0.5, 0.24, 0.29)
    box(ax, 0.48, 0.62, 0.2, 0.18, 'shared LSTM\nprocess-1 (self)', frozen=other_frozen)
    box(ax, 0.48, 0.2, 0.2, 0.18, 'shared LSTM\nprocess-2 (other)', frozen=other_frozen)
    arrow(ax, 0.40, 0.71, 0.48, 0.71, 'masked'); arrow(ax, 0.40, 0.29, 0.48, 0.29, 'masked')
    ax.text(0.58, 0.88, p1_text, fontsize=6, ha='center', color='tab:red')
    arrow(ax, 0.58, 0.86, 0.58, 0.80)
    if stage2:
        box(ax, 0.30, 0.02, 0.16, 0.1, 'other module G\n(trained)', color='#fde0c5')
        arrow(ax, 0.32, 0.24, 0.36, 0.12, 'unmasked', off=(-0.06, 0))
        arrow(ax, 0.46, 0.07, 0.58, 0.20, p2_text, off=(0.08, -0.04))
    else:
        ax.text(0.58, 0.07, p2_text, fontsize=6, ha='center', color='tab:red'); arrow(ax, 0.58, 0.1, 0.58, 0.2)
    box(ax, 0.74, 0.42, 0.11, 0.16, 'integration', frozen=other_frozen)
    arrow(ax, 0.68, 0.71, 0.74, 0.54); arrow(ax, 0.68, 0.29, 0.74, 0.46)
    box(ax, 0.88, 0.42, 0.1, 0.16, 'Decoder\n-> v_{t+1}', frozen=other_frozen)
    arrow(ax, 0.85, 0.5, 0.88, 0.5)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    panel(axes[0, 0], 'v11 stage 1 (all modules trained from scratch)',
          'pleasure P_t (8)', 'zeros (8)', False, False, False)
    panel(axes[0, 1], 'v11 stage 2 (only G trained)',
          'pleasure P_t (8)', 'Q2 (8)', True, True, True)
    panel(axes[1, 0], 'v12 stage 1 (encoders = frozen v4; rest trained)',
          '[motion m_t (2, hidden in 5-15 step intervals),\n P_t (8, held while motion hidden), flag (1)]',
          'zeros (11)', True, False, False)
    panel(axes[1, 1], 'v12 stage 2 (only G trained)',
          '[m_t (2, visible), P_t (8), flag 0]', '[0, 0, Q2 (8), flag 1]', True, True, True)
    fig.suptitle('Networks (hatched = frozen). Prediction target: the next view v_{t+1} only.', fontsize=10)
    fig.tight_layout()
    out = os.path.join(RES, 'v12', 'figs'); os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, 'network_v11_v12.png'), dpi=130)
    print('saved', out)


if __name__ == '__main__':
    main()
