"""
v14-A smoke-test figure (docs/v14_instructions.md Sec.4.1 item 3; not used for any judgement).
From data/result/v14/<sub>/<tag>_pref.npz (eval_inverse_v14.py --keep_pred):
  (a) A-2 position read from the predicted image, cp minus cm, projected on the axis from the bins-0,1
      landmark side to the bins-6,7 landmark side (+y), mean over episodes and per episode;
  (b) cumulative self-vision L1 difference (cm - cp) vs T per episode, coloured by the correct candidate;
  (c) predicted images under cp / cm and the true view for one episode at a few steps.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import h5py  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
from eval_s2_v11 import load_set  # noqa

C_PLUS, C_MINUS = '#2a78d6', '#eb6834'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sub', default='smoke')
    ap.add_argument('--tag', default='v12_mixed_0_ep200')
    ap.add_argument('--ep', type=int, default=0)
    args = ap.parse_args()
    base = os.path.join(_PI, 'data', 'result', 'v14', args.sub)
    z = np.load(os.path.join(base, f'{args.tag}_pref.npz'))
    cs = z['correct_sign']
    N = len(cs)
    dy = z['est_cp'][..., 1] - z['est_cm'][..., 1]          # (N, 100), target frames 1..100
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 1, 1])
    ax = fig.add_subplot(gs[0, :2])
    t = np.arange(1, 101)
    for i in range(N):
        ax.plot(t, dy[i], color='0.75', lw=0.8)
    ax.plot(t, np.nanmean(dy, 0), color='k', lw=2, label='mean over episodes')
    ax.axhline(0, color='0.4', lw=0.8)
    ax.set_xlabel('target frame t')
    ax.set_ylabel('A-2 y (cp) - A-2 y (cm)\n(+ = toward bins 6,7 side)')
    ax.set_title(f'(a) A-2 read from predicted image, cp - cm\n(N={N}, read in '
                 f'{np.isfinite(dy).mean():.2f} of frames)', fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax = fig.add_subplot(gs[0, 2:])
    cum = np.cumsum(z['L1_cm'] - z['L1_cp'], 1)
    for i in range(N):
        ax.plot(t, cum[i], color=C_PLUS if cs[i] > 0 else C_MINUS, lw=1)
    ax.plot([], [], color=C_PLUS, label='correct = cp')
    ax.plot([], [], color=C_MINUS, label='correct = cm')
    ax.axhline(0, color='0.4', lw=0.8)
    for T in (5, 10, 20, 30, 50, 100):
        ax.axvline(T, color='0.85', lw=0.6, zorder=0)
    ax.set_xlabel('T')
    ax.set_ylabel('cumulative L1 (cm) - (cp)\n(+ = cp chosen)')
    acc = {T: float(((cum[:, T - 1] > 0) == (cs > 0)).mean()) for T in (5, 10, 20, 50, 100)}
    ax.set_title('(b) cumulative L1 difference per episode\nraw acc ' +
                 ', '.join(f'T{T} {a:.2f}' for T, a in acc.items()), fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    d = load_set('pref')
    tv = h5py.File(d['path'], 'r')[d['split']]['self_vision']
    e = args.ep
    steps = (5, 20, 40, 70)
    for j, s in enumerate(steps):
        rows = ((z['pred_cp'][e, s - 1].astype(np.float32), 'predicted, cp'),
                (z['pred_cm'][e, s - 1].astype(np.float32), 'predicted, cm'),
                (tv[e, s], 'true view'))
        for r, (img, nm) in enumerate(rows):
            a = fig.add_subplot(gs[1 + r, j])
            a.imshow(img, interpolation='nearest')
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f'{nm}, frame {s}', fontsize=8)
    lay, lk = int(z['layout'][e]), int(z['a2_liked'][e])
    fig.suptitle(f'v14-A smoke test ({args.tag}, ② test first {N} ep; not a judgement).  '
                 f'Example episode {e}: layout {lay}, A-2 likes {"Red" if lk else "Green"}, correct = '
                 f'{"cp" if cs[e] > 0 else "cm"}', fontsize=10)
    fig.tight_layout()
    out = os.path.join(base, f'smoke_{args.tag}.png')
    fig.savefig(out, dpi=110)
    print(out, 'mean dy (t>=5)', float(np.nanmean(dy[:, 4:])), acc)


if __name__ == '__main__':
    main()
