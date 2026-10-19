"""
v12 figures for G5a / G5b (instructions Sec.5.2) from eval_g5_v12.py outputs
(data/result/v12/<sub>/g5_{random,mixed}_ep<e>.json):
  g5a_curves.png : per step k after the freeze, mean projection of the A-2 position read from the
                   predicted image on the liked-colour direction and on the other three directions
                   (relative to A-2's position at the freeze), both conditions, sets 1 and 2;
  g5b_bins.png   : per bin k, projection on u_k of the +a shift, the -a shift and their difference
                   (10 steps), both conditions; plus the mean shift vectors.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v12 import RES  # noqa
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE))))
from v11core.pleasure import BIN_UNIT  # noqa

CC = {'random': 'tab:gray', 'mixed': 'tab:red'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sub', required=True)
    ap.add_argument('--epoch', type=int, default=150)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    d = os.path.join(RES, 'v12', args.sub)
    G = {c: json.load(open(os.path.join(d, f'g5_{c}_ep{args.epoch}.json'))) for c in ('random', 'mixed')}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, k, nm in zip(axes, ('green', 'pref'), ('set 1 (green-liker)', 'set 2 (preference per episode)')):
        for c in ('random', 'mixed'):
            r = G[c]['G5a'][f'{k}_K15_freeze']
            K = len(r['curve_liked'])
            ax.plot(range(1, K + 1), r['curve_liked'], color=CC[c], lw=2, label=f'{c}: liked colour')
            ax.plot(range(1, K + 1), r['curve_others'], color=CC[c], lw=1, ls='--', label=f'{c}: other 3 (mean)')
            ax.text(0.02, 0.95 - 0.07 * (c == 'mixed'), f"{c}: g_other {r['g_other']['mean']:+.3f} [{r['g_other']['ci95'][0]:+.3f}, {r['g_other']['ci95'][1]:+.3f}], read {r['frac_read']:.2f}",
                    transform=ax.transAxes, fontsize=7, color=CC[c])
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('step k after freezing P_A2 (process-2 = motion 0, flag 1)')
        ax.set_ylabel('A-2 displacement projected (arena units)')
        ax.set_title(f'G5a other-side accumulation, {nm}', fontsize=9); ax.legend(fontsize=7, loc='lower left')
    fig.tight_layout(); fig.savefig(os.path.join(d, f'g5a_curves{args.tag}.png'), dpi=120); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    w = 0.13
    for ci_, c in enumerate(('random', 'mixed')):
        b = G[c]['G5b']
        for j, (key, lab, a) in enumerate((('own_p_per_bin', '+a', 0.5), ('own_m_per_bin', '-a', 0.3), ('pm_own_per_bin', '(+a)-(-a)', 1.0))):
            axes[0].bar(np.arange(8) + (ci_ * 3 + j - 2.5) * w, b[key], w, color=CC[c], alpha=a, label=f'{c} {lab}')
    axes[0].axhline(0, color='k', lw=0.6); axes[0].set_xticks(range(8)); axes[0].set_xlabel('bin k of the manipulation')
    axes[0].set_ylabel('shift of A-2 projected on u_k after 10 steps'); axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_title(f"G5b per bin; main = mean of (+a)-(-a): random {G['random']['G5b']['main_pm_own']['mean']:+.3f}, mixed {G['mixed']['G5b']['main_pm_own']['mean']:+.3f}", fontsize=8)
    for ax, c in zip(axes[1:], ('random', 'mixed')):
        b = G[c]['G5b']
        P, M = np.array(b['mean_vec_p']), np.array(b['mean_vec_m'])
        lim = max(0.5, np.nanmax(np.abs(np.concatenate([P, M]))) * 1.3)
        for k in range(8):
            u = BIN_UNIT[k] * lim * 0.95
            ax.plot([0, u[0]], [0, u[1]], color='0.85', lw=0.8); ax.text(u[0], u[1], str(k), fontsize=7, color='0.5')
            ax.arrow(0, 0, P[k, 0], P[k, 1], color='tab:red', width=lim * 0.006, length_includes_head=True)
            ax.arrow(0, 0, M[k, 0], M[k, 1], color='tab:green', width=lim * 0.006, length_includes_head=True)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
        ax.set_title(f'{c}: mean A-2 shift after 10 steps (red +a, green -a in bin k; grey = bin directions)', fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(d, f'g5b_bins{args.tag}.png'), dpi=120); plt.close(fig)
    print('saved', d)


if __name__ == '__main__':
    main()
