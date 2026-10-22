"""
v14-A figures (docs/v14_experiment_log.md Sec.3 E) from the aggregate JSONs of aggregate_inverse_v14.py:
  inverse_ba_vs_T.png : balanced accuracy vs T, one panel per pair (v12 s0..s3, v11 s0, s2), main set (pref, L1);
                        experienced / no-experience (mean over late-5, bars = min..max over the 5 checkpoints),
                        reference (true P_A2) for the experienced model, chance 0.5;
  inverse_diff_dist.png : per-episode cumulative-error difference (wrong - correct candidate) at T = 50, ep200.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
BASE = os.path.join(_PI, 'data', 'result', 'v14', 'inverse')
TS = (5, 10, 20, 30, 50, 100)
C_EXP, C_NO, C_REF = '#2a78d6', '#eb6834', '0.55'


def pairs():
    out = []
    for name in ('aggregate_v12_s0s1s2s3', 'aggregate_v11_s0s2'):
        d = json.load(open(os.path.join(BASE, name + '.json')))['detail']
        out += [(k, v) for k, v in d.items()]
    return out


def series(v, key, s='pref', q='L1'):
    a = np.array([[p[key] for p in v[f'{s}_{q}_T{T}']['points'].values()] for T in TS])   # (T, 5)
    return a.mean(1), a.min(1), a.max(1)


def main(s='pref', q='L1'):
    P = pairs()
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, (k, v) in zip(axs.ravel(), P):
        x = np.arange(len(TS))
        for key, col, lab in (('ba_exp', C_EXP, 'experienced'), ('ba_noexp', C_NO, 'no experience'),
                              ('ref_ba_exp', C_REF, 'reference: true P_A2 (exp. model)')):
            m, lo, hi = series(v, key, s, q)
            ax.errorbar(x, m, yerr=[m - lo, hi - m], color=col, lw=2, marker='o', ms=5, capsize=3, label=lab,
                        ls='--' if key.startswith('ref') else '-')
        ax.axhline(0.5, color='0.3', lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([str(T) for T in TS])
        fam, _, sd = k.split(',')[0].split(':')
        npass = v[f'{s}_{q}_T50']['n_pass']
        ax.set_title(f'{fam} pair s{sd}  (T=50 passing points {npass}/5)', fontsize=10)
        ax.set_xlabel('T (frames accumulated)')
    for ax in axs[:, 0]:
        ax.set_ylabel('balanced accuracy')
    axs[0, 0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle(f'v14-A: choosing c+ / c- by the cumulative prediction error ({s} test, {q}); '
                 'mean over late-5, bars = range over the 5 checkpoints', fontsize=11)
    fig.tight_layout()
    out = os.path.join(BASE, f'inverse_ba_vs_T_{s}_{q}.png')
    fig.savefig(out, dpi=110)
    print(out)
    # distribution of the cumulative-error difference at T = 50, ep200
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    for ax, (k, v) in zip(axs.ravel(), P):
        for spec, col, lab in ((k.split(',')[0], C_EXP, 'experienced'), (k.split(',')[1], C_NO, 'no experience')):
            z = np.load(os.path.join(BASE, spec.replace(':', '_') + f'_ep200_{s}.npz'))
            ep, em = z[f'{q}_cp'][:, :50].sum(1), z[f'{q}_cm'][:, :50].sum(1)
            d = np.where(z['correct_sign'] > 0, em - ep, ep - em)
            ax.hist(d, bins=40, histtype='step', color=col, lw=1.5, label=f'{lab} (median {np.median(d):+.2f})')
        ax.axvline(0, color='0.3', lw=0.8)
        fam, _, sd = k.split(',')[0].split(':')
        ax.set_title(f'{fam} pair s{sd}, ep200', fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.supxlabel(f'cumulative {q} error, wrong - correct candidate (T=50; > 0 = correct chosen)', fontsize=10)
    fig.tight_layout()
    out = os.path.join(BASE, f'inverse_diff_dist_{s}_{q}.png')
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == '__main__':
    main()
    main('green')
    main('pref', 'FP')
