"""
Proposal for paper v3 figure 3 (2026-11-01): the main comparison only. For G5b and the sign-following part (green,
mixed), the per-pair difference 'sign-randomised - normal' at 50% (meets the tracking premise) and at 80% (does not).
Same saved values as v3_other (fig_v3_paper.load; no new computation). One point per pair (pairs 1-3 grey circles,
pairs 4-6 orange squares), black square with bar = mean of the 6 pairs and t 95% CI (df 5); the means and CIs are the
values of table tab:self (log Sec.34.8, Sec.35.11; compared by hand on 2026-11-01).
-> paper/figs/v3_other_diff.{pdf,png}, v3_other_diff_values.json
"""

import json
import os

import paper_style as ps
from fig_v3_paper import OTHER, TEXTWIDTH_IN, late5, load, mean_ci


def main():
    ps.setup()
    plt = ps.plt
    V = load()
    vals = {}
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.5), sharey=False)
    for ax, (q, title) in zip(axes, OTHER):
        out = {}
        for x, amt in ((0, '50'), (1, '80')):
            d = [late5(V[s]['sr' + amt][q]) - late5(V[s][amt][q]) for s in range(6)]
            for s, v in enumerate(d):
                col = ps.C_MAIN if s <= 2 else ps.C_EXTRA
                ax.plot(x - 0.12 + 0.05 * (s % 3), v, 'o' if s <= 2 else 's', color=col, ms=3.2,
                        mfc=col if s <= 2 else 'white', mew=0.8, zorder=2)
            m, lo, hi = mean_ci(d)
            ax.errorbar(x + 0.2, m, yerr=[[m - lo], [hi - m]], fmt='s', color='k', ms=3.6, lw=1.5, capsize=2.5, zorder=3)
            out[amt] = {'mean_ci': [m, lo, hi], 'pairs': {ps.pair_label(s): d[s] for s in range(6)}}
        vals[q] = out
        ax.axhline(0, color='0.6', lw=0.6, zorder=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['経験50%\n（前提を満たす）', '経験80%\n（満たさない）'])
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(title, loc='left', fontsize=8.5)
    axes[0].set_ylabel('符号無作為 $-$ 通常')
    h = [plt.Line2D([], [], color=ps.C_MAIN, marker='o', ls='', ms=3.2, label='組1〜3'),
         plt.Line2D([], [], color=ps.C_EXTRA, marker='s', mfc='white', ls='', ms=3.2, label='組4〜6'),
         plt.Line2D([], [], color='k', marker='s', ms=3.6, lw=1.5, label='6組の平均と95%信頼区間（$t$，自由度5）')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.0)
    ps.save(fig, 'v3_other_diff')
    json.dump(vals, open(os.path.join(ps.FIGS, 'v3_other_diff_values.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
