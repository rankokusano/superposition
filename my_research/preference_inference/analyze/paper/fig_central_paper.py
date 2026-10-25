"""
Paper figure 3 'central' (paper round 1; documents only, CPU): the v13 central figure (analyze/v13/fig_central_v13.py)
redrawn for the paper with Japanese labels and no version numbers. Same data and averaging:
pair values (late-5 means) from data/result/v13/extra/across_pairs.json; the across-pair mean and 95% CI (t, df = 5);
the no-sharing condition (80%, pairs 1-3) from the NS per-episode files via fig_central_v13.ns_values.
Writes paper/figs/central.{pdf,png} and paper/figs/central_values.json (the plotted numbers).
"""

import json
import os
import sys

import numpy as np
from scipy import stats

from paper_style import (C_EXTRA, C_MAIN, C_NS, FIGS, M_NS, PI, RES, pair_label, plt, save, setup)

sys.path.insert(0, os.path.join(PI, 'analyze', 'v13'))
from fig_central_v13 import ns_values  # noqa

PANELS = (('G3', '(a) 自己側の積み上がり $g$'), ('sign_green', '(b) 符号に従う分（緑好みデータ）'),
          ('sign_pref', '(c) 符号に従う分（好み混合データ）'), ('G5b', '(d) 他者側の符号応答 G5b'))
CONDS = ('none', '50', '80')
XT = ['経験なし', '経験50%', '経験80%']


def main():
    setup()
    pp = json.load(open(os.path.join(RES, 'v13', 'extra', 'across_pairs.json')))['per_pair']
    seeds = sorted(pp, key=lambda k: int(k[1:]))
    ns = {s: ns_values(s) for s in (0, 1, 2)}
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6))
    vals = {'pairs': {}, 'mean_ci': {}, 'ns': {s: ns[s] for s in ns}}
    for ax, (q, title) in zip(axes.flat, PANELS):
        for sk in seeds:
            s = int(sk[1:])
            main_pair = s <= 2
            ax.plot(range(3), [pp[sk][q][c] for c in CONDS], ls='-' if main_pair else '--', marker='o', ms=2.5,
                    lw=0.8, color=C_MAIN if main_pair else C_EXTRA,
                    label=('組1〜3（本番の組）' if s == 0 else '組4〜6（予備・追加の組）' if s == 3 else None))
        M = np.array([[pp[sk][q][c] for c in CONDS] for sk in seeds])
        mu = M.mean(0)
        h = stats.t.ppf(0.975, len(seeds) - 1) * M.std(0, ddof=1) / np.sqrt(len(seeds))
        vals['mean_ci'][q] = {c: [float(mu[i]), float(mu[i] - h[i]), float(mu[i] + h[i])] for i, c in enumerate(CONDS)}
        vals['pairs'][q] = {pair_label(int(sk[1:])): {c: pp[sk][q][c] for c in CONDS} for sk in seeds}
        ax.errorbar(np.arange(3) + 0.1, mu, yerr=h, color='k', lw=1.8, capsize=3, marker='s', ms=4,
                    label='6組の平均と95%信頼区間（$t$，自由度5）')
        ax.plot([2.32] * 3, [ns[s][q] for s in (0, 1, 2)], M_NS, mfc='white', mec=C_NS, ms=5, mew=1.3,
                label='非共有条件（経験80%，組1〜3）')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(XT)
        ax.set_xlim(-0.3, 2.55)
        ax.set_title(title, loc='left')
        ax.set_ylabel('最終5点の平均')
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=1.2, w_pad=1.5)
    save(fig, 'central')
    json.dump(vals, open(os.path.join(FIGS, 'central_values.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
