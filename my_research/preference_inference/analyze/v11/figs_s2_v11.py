"""
v11 stage-2 figures (docs/v11_experiment_log.md Sec.7.49-7.55), from saved outputs only:
  fig_s2_primary.png : T2 (setting 1) and T5 (setting 2) per G checkpoint, mixed vs random, D6 level;
  fig_s2_t5_curve.png: d_T5 by time window (late-5 mean), with D6;
  fig_s2_t3.png      : T3 self-vision L1 and other feature-prediction MSE per Q2 condition;
  fig_s2_q2_bins.png : per-bin mean and std of generated Q2 vs P_A2 (ep200, v11_s2t_green).
Writes under data/result/v11/stage2/figs/.
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import torch  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_s2_v11 import OUT, load_set, run_q2, true_pleasure  # noqa
from eval_trial_v11 import load_model  # noqa

FIG = os.path.join(os.path.dirname(OUT), 'figs')
EPS = [160, 170, 180, 190, 200]
COL = {'random': 'tab:gray', 'mixed': 'tab:red'}


def main():
    os.makedirs(FIG, exist_ok=True)
    A = json.load(open(os.path.join(OUT, 'aggregate_primary.json')))
    d6 = A['d6_PA2']
    # primary
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (key, title, ref) in zip(axes, (('green:sT2_t50', 'Setting 1: T2 s_T2 (t=50)', d6['T2_mean']['t50']),
                                            ('pref:dT5', 'Setting 2: T5 d_T5', d6['T5_mean']))):
        for s, mk in ((0, 'o'), (2, 's')):
            for c in ('random', 'mixed'):
                y = [x[c] for x in A[key][f's{s}']['per_ckpt']]
                ax.plot(EPS, y, marker=mk, color=COL[c], label=f'{c} s{s}', ls='-' if s == 0 else '--')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title(f'{title}\n(D6 = P_A2 itself: {ref:+.2f}, off scale)')
        ax.set_xlabel('G checkpoint (epoch)'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_s2_primary.png'), dpi=150); plt.close(fig)
    # T5 curve
    S = A['secondary_late5_mean_min_max']
    win = ['20-40', '40-60', '60-80', '80-100']
    fig, ax = plt.subplots(figsize=(5, 3.8))
    for st, lsty in (('pref', '-'), ('green', ':')):
        for s, mk in ((0, 'o'), (2, 's')):
            for c in ('random', 'mixed'):
                ax.plot(range(4), [S[f'{st}_{c}_s{s}'][f'T5_t/{w}'][0] for w in win], marker=mk, ls=lsty, color=COL[c],
                        label=f'{"2" if st == "pref" else "1"} {c} s{s}')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(win); ax.set_xlabel('t window')
    ax.set_title(f'd_T5 by time (late-5 mean); D6 {min(d6["T5_by_t_bins"].values()):+.2f}..{max(d6["T5_by_t_bins"].values()):+.2f}')
    ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_s2_t5_curve.png'), dpi=150); plt.close(fig)
    # T3
    T = json.load(open(os.path.join(os.path.dirname(OUT), 't1t3', 'aggregate_t1t3.json')))
    conds = [('zero', 'zero'), ('gen', 'generated'), ('d6', 'D6 (P_A2)'), ('proxy', 'position proxy')]
    runs = [(st, c, s) for st in ('green', 'pref') for s in (0, 2) for c in ('random', 'mixed')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (lk, ttl) in zip(axes, (('self_L1', 'self-vision L1 / pixel'), ('other_fp_mse', 'other feature-prediction MSE'))):
        w = 0.2
        for i, (cd, cn) in enumerate(conds):
            ax.bar(np.arange(len(runs)) + (i - 1.5) * w, [T[f'{st}_{c}_s{s}'][f'green/{cd}/{lk}'][0] for st, c, s in runs], w, label=cn)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels([f'{"1" if st == "green" else "2"} {c[:3]} s{s}' for st, c, s in runs], rotation=45, fontsize=7)
        ax.set_title(f'T3: {ttl} (v11_s2t_green, late-5)'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_s2_t3.png'), dpi=150); plt.close(fig)
    # Q2 per bin
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    d = load_set('green')
    P = true_pleasure('green', d)[:, 1:100].reshape(-1, 8)
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharey=True)
    for ax, (st, c, s) in zip(axes.ravel(), runs):
        m, _ = load_model(f'v11_s2_{st}_{c}_s{s}', s, 200, dev)
        Q, _ = run_q2(m, d['path'], d['split'], dev)
        Q = Q[:, 1:100].reshape(-1, 8)
        ax.errorbar(np.arange(8) - 0.1, Q.mean(0), Q.std(0), fmt='o', color=COL[c], label='Q2')
        ax.errorbar(np.arange(8) + 0.1, P.mean(0), P.std(0), fmt='s', color='tab:green', label='P_A2')
        ax.axhline(0, color='k', lw=0.6); ax.axvspan(1.5, 5.5, color='0.9', zorder=0)
        ax.set_title(f'{"1" if st == "green" else "2"} {c} s{s} ep200', fontsize=9); ax.set_xticks(range(8))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Generated Q2 vs P_A2 per bin (mean +- std, v11_s2t_green; shaded: bins 2-5, P_A2 = 0)')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_s2_q2_bins.png'), dpi=150); plt.close(fig)
    print('saved', FIG)


if __name__ == '__main__':
    main()
