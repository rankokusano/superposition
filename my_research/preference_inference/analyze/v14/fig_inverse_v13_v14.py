"""
v14-A on v13 figures (docs/v14_experiment_log.md Sec.27):
  inverse_v13_ba_vs_T.png : BA vs T (pref test, self-vision L1, l1 late-5 mean, bars = range over the 5 checkpoints)
                            per pair: v13 experienced, v12 experienced (same seed), v12 no-experience;
  inverse_v13_follow_pref.png : share of choices following preference among episodes where preference and A-2's
                            direction disagree (95% CI), FP (mse late-5 and l1 point) and L1 (l1 point), v13 exp,
                            v12 exp, v12 no-exp; dashed line 0.5.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
B = os.path.join(_PI, 'data', 'result', 'v14', 'inverse')
TS = (5, 10, 20, 30, 50, 100)
EP = ('160', '170', '180', '190', '200')
C13, C12, CNO = '#2a78d6', '#1baf7a', '#eb6834'


def curve(detail, pair, key):
    a = np.array([[detail[pair][f'pref_L1_T{T}']['points'][e][key] for e in EP] for T in TS])
    return a.mean(1), a.min(1), a.max(1)


def main():
    v13 = json.load(open(os.path.join(B, 'aggregate_v13_s0s1s2.json')))['detail']
    v12 = json.load(open(os.path.join(B, 'aggregate_v12_s0s1s2.json')))['detail']
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    x = np.arange(len(TS))
    for s, ax in enumerate(axs):
        for d, pr, key, col, lab in ((v13, f'v13:mixed:{s},v12:random:{s}', 'ba_exp', C13, 'v13 experienced (80%)'),
                                     (v12, f'v12:mixed:{s},v12:random:{s}', 'ba_exp', C12, 'v12 experienced (50%)'),
                                     (v12, f'v12:mixed:{s},v12:random:{s}', 'ba_noexp', CNO, 'no experience (v12)')):
            m, lo, hi = curve(d, pr, key)
            ax.errorbar(x, m, yerr=[m - lo, hi - m], color=col, lw=2, marker='o', ms=5, capsize=3, label=lab)
        ax.axhline(0.5, color='0.3', lw=0.8)
        ax.axvline(x[TS.index(50)], color='0.85', lw=6, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels([str(T) for T in TS]); ax.set_xlabel('T (frames accumulated)')
        n13 = v13[f'v13:mixed:{s},v12:random:{s}']['pref_L1_T50']['n_pass']
        ax.set_title(f'pair s{s}  (v13 passing points at T=50: {n13}/5)', fontsize=10)
    axs[0].set_ylabel('balanced accuracy')
    axs[0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('v14-A: choosing c+ / c- by the cumulative self-vision error (pref test); '
                 'mean over l1 late-5, bars = range; grey band = registered T', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(B, 'inverse_v13_ba_vs_T.png'), dpi=110)
    # follow-preference among disagreeing episodes
    dm = json.load(open(os.path.join(B, 'fp_direction_v13mse.json')))
    dl = json.load(open(os.path.join(B, 'fp_direction_v13l1.json')))
    d12 = json.load(open(os.path.join(B, 'fp_direction.json')))   # v12 (Sec.19 keys: <cond>_s<s>|...)
    panels = (('FP, mse late-5 (registered count)', 'mse_FP'), ('FP, l1 point', 'l1_FP'), ('self-vision L1, l1 late-5 / point', 'l1_L1'))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (title, kind) in zip(axs, panels):
        for s in range(3):
            rows = []
            if kind == 'mse_FP':
                rows = [('v13 exp', C13, [dm[f'v13mse_mixed_{s}|FP|all|ep{e}'] for e in EP]),
                        ('no exp', CNO, [dm[f'v12mse_random_{s}|FP|all|ep{e}'] for e in EP])]
            elif kind == 'l1_FP':
                rows = [('v13 exp', C13, [dl[f'v13_mixed_{s}|FP|all|ep200']]),
                        ('v12 exp', C12, [d12[f'mixed_s{s}|FP|all|ep200']]),
                        ('no exp', CNO, [dl[f'v12_random_{s}|FP|all|ep200']])]
            else:
                rows = [('v13 exp', C13, [dl[f'v13_mixed_{s}|L1|all|ep200']]),
                        ('v12 exp', C12, [d12[f'mixed_s{s}|L1|all|ep{e}'] for e in EP]),
                        ('no exp', CNO, [dl[f'v12_random_{s}|L1|all|ep200']])]
            for j, (lab, col, pts) in enumerate(rows):
                xs = s + (j - (len(rows) - 1) / 2) * 0.22
                for i, p in enumerate(pts):
                    xx = xs + (i - (len(pts) - 1) / 2) * 0.035
                    ax.errorbar(xx, p['disagree_follow_pref'], yerr=[[p['disagree_follow_pref'] - p['disagree_follow_pref_ci'][0]],
                                [p['disagree_follow_pref_ci'][1] - p['disagree_follow_pref']]], color=col, marker='o', ms=4,
                                capsize=2, lw=1, label=lab if (s == 0 and i == 0) else None)
        ax.axhline(0.5, color='0.3', lw=0.8, ls='--')
        ax.set_xticks(range(3)); ax.set_xticklabels([f'pair s{s}' for s in range(3)])
        ax.set_title(title, fontsize=10)
    axs[0].set_ylabel('share following preference')
    axs[1].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('Does the choice follow A-2\'s preference or its direction? Episodes where they disagree (n = 88), '
                 '95% CI; > 0.5 = preference', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(B, 'inverse_v13_follow_pref.png'), dpi=110)
    print('ok')


if __name__ == '__main__':
    main()
