"""
Central figure (docs/v13_experiment_log.md Sec.34; documents only, CPU): x = no experience / 50% / 80%, y = late-5 mean.
Panels: G3 (self side), sign-following part (1), sign-following part (2), G5b. Each pair s0-s5 as thin lines (main pairs
s0-s2 solid, extra pairs s3-s5 dashed), the across-pair mean and 95% CI (t, df = 5) as a thick bar, and v13-NS
(80%, no sharing, s0-s2) as separate markers next to 80%. Pair values from extra/across_pairs.json
(analyze/v13/across_pairs_v13.py); NS values from the NS per-episode files with the same late-5 averaging.
Writes data/result/v13/compare_all/figs/central_v13.png.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

EPS = (160, 170, 180, 190, 200)
N = lambda *a: np.load(os.path.join(RES, *a))  # noqa


def ns_values(s):
    v = {}
    for e in EPS:
        p = N('v13', 'ns', 's1eval', f's{s}', f'per_ep{e}.npz')
        g = N('v13', 'ns', 'g5', f's{s}', f'g5_per_ep{e}.npz')
        d = N('v13', 'post', 'g5a_decomp_ns', f's{s}', f'decomp_mixed80_ep{e}.npz')
        v.setdefault('G3', []).append(np.nanmean(p['mixed__freeze_K15_hold']))
        v.setdefault('G5b', []).append(np.nanmean(g['mixed__G5b']))
        for k in ('green', 'pref'):
            v.setdefault(f'sign_{k}', []).append(np.nanmean((g[f'mixed__G5a_{k}_K15_freeze'] - d[f'{k}_flip_g']) / 2))
    return {q: float(np.mean(x)) for q, x in v.items()}


def main():
    pp = json.load(open(os.path.join(RES, 'v13', 'extra', 'across_pairs.json')))['per_pair']
    seeds = sorted(pp, key=lambda k: int(k[1:]))
    ns = {s: ns_values(s) for s in (0, 1, 2)}
    panels = (('G3', 'Self side: G3 accumulation g'), ('sign_green', 'Other side: sign-following part, set 1'),
              ('sign_pref', 'Other side: sign-following part, set 2'), ('G5b', 'Other side: G5b'))
    conds = ('none', '50', '80')
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (q, title) in zip(axes, panels):
        for sk in seeds:
            main_pair = int(sk[1:]) <= 2
            ax.plot(range(3), [pp[sk][q][c] for c in conds], ls='-' if main_pair else '--', marker='o', ms=3.5, lw=0.9,
                    color='0.35' if main_pair else 'tab:orange', alpha=0.85,
                    label=('main pairs s0-s2' if sk == 's0' else 'extra pairs s3-s5' if sk == 's3' else None))
        M = np.array([[pp[sk][q][c] for c in conds] for sk in seeds])
        mu = M.mean(0)
        h = stats.t.ppf(0.975, len(seeds) - 1) * M.std(0, ddof=1) / np.sqrt(len(seeds))
        ax.errorbar(np.arange(3) + 0.08, mu, yerr=h, color='k', lw=2.4, capsize=5, marker='s', ms=6,
                    label='mean over 6 pairs, 95% CI (t, df=5)')
        nv = [ns[s][q] for s in (0, 1, 2)]
        ax.plot([2.3] * 3, nv, 'D', mfc='white', mec='tab:blue', ms=6, mew=1.5, label='v13-NS 80%, no sharing (s0-s2)')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['no experience', '50%', '80%'])
        ax.set_xlim(-0.3, 2.55)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('late-5 mean' if q == 'G3' else '')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle('Expectation from own experience grows with its amount on the self side and, through the shared module, '
                 'on the other side (A-1 follows its preference on 0 / 50 / 80% of steps)', fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(RES, 'v13', 'compare_all', 'figs', 'central_v13.png')
    fig.savefig(out, dpi=150)
    print(out, {s: {q: round(v, 3) for q, v in ns[s].items()} for s in ns})


if __name__ == '__main__':
    main()
