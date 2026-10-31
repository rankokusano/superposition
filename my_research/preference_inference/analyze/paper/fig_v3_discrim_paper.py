"""
Paper v3 figure for chapter 6 (task 2, V-dir stage B; paper/draft_v3, 2026-10-31). No new computation: the saved
pre-registered aggregation only (data/result/paper_extra/Vdir_B/analysis/summary.json; vdirb_analysis.py b500bac,
results 02618a4, paper/check/vdir_stageB.md Sec.3, Sec.9).
  (a) main regression (c+-, FP, frames 1-50, tau = 1.0, with initial positions, 1088 episodes with a direction):
      b_pref and b_dir for no experience / 50% / sign-randomised (50%);
  (b) two-choice BA with the preference label (1200 episodes) and the direction label (1088 episodes);
  (c) preference accuracy on the 280 episodes where preference and direction disagree (0.5 line).
Pairs 1-3 grey circles, pairs 4-6 orange squares, one thin line per pair; black square with bar = mean of the 6 pairs
and t 95% CI (df 5), checked against the saved t CI. Each per-pair value = mean over the 5 late checkpoints.
-> paper/figs/v3_discrim.{pdf,png}, v3_discrim_values.json
"""

import json
import os

import numpy as np

import paper_style as ps
from fig_v3_paper import TEXTWIDTH_IN, mean_ci

SUMMARY = os.path.join(ps.PI, 'data', 'result', 'paper_extra', 'Vdir_B', 'analysis', 'summary.json')
CONDS = ('none', 'p50', 'sr50')
XT = ['経験\nなし', '経験\n50%', '符号\n無作為']


def draw(ax, groups, xt, ref):
    """groups: list of (x positions, {pair: [values]}, saved t summaries). Thin line per pair, mean and CI."""
    out = {}
    for xs, per, saved in groups:
        for s, ys in per.items():
            col = ps.C_MAIN if s <= 2 else ps.C_EXTRA
            ax.plot(xs, ys, '-', marker='o' if s <= 2 else 's', color=col, ms=3.0, lw=0.8,
                    mfc=col if s <= 2 else 'white', mew=0.8, zorder=2)
        for i, x in enumerate(xs):
            m, lo, hi = mean_ci([per[s][i] for s in per])
            t = saved[i]
            assert abs(m - t['mean']) < 1e-12 and abs(lo - t['ci95'][0]) < 1e-12 and abs(hi - t['ci95'][1]) < 1e-12, (x, m, t)
            ax.errorbar(x + 0.12, m, yerr=[[m - lo], [hi - m]], fmt='s', color='k', ms=3.6, lw=1.5, capsize=2.5, zorder=3)
            out[str(x)] = [m, lo, hi]
    ax.axhline(ref, color='0.6', lw=0.6, ls='-' if ref == 0 else '--', zorder=0)
    ax.set_xticks([x for xs, _, _ in groups for x in xs])
    ax.set_xticklabels(xt)
    ax.tick_params(axis='x', labelsize=7)
    return out


def main():
    ps.setup()
    plt = ps.plt
    S = json.load(open(SUMMARY))
    coef = S['coef']
    sec = S['secondary']
    vals = {}

    # (a) and (b) have six positions each; side by side with (c) the 7 pt tick labels touch at the text width,
    # so (a)(b) are the top row and (c) with the legend the bottom row.
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 4.9))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1])
    axes = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[0, 3:]), fig.add_subplot(gs[1, :2])]

    # (a) b_pref and b_dir of the main regression
    ax = axes[0]
    groups = []
    for x0, term in ((0, 'pref'), (3.5, 'dir')):
        es = [coef[f'cpm|FP|1-50|1.0|main|{c}'] for c in CONDS]
        assert all(e['n_used'] == 1088 for e in es)
        per = {s: [e['per_pair'][term][s] for e in es] for s in range(6)}
        groups.append(([x0, x0 + 1, x0 + 2], per, [e['t'][term] for e in es]))
        vals[f'b_{term}_pairs'] = {ps.pair_label(s): per[s] for s in range(6)}
    vals['b'] = draw(ax, groups, XT * 2, 0)
    ax.axvline(2.75, color='0.8', lw=0.6, zorder=0)
    ax.set_xlim(-0.4, 6.0)
    for x, txt in ((1.06, '好みの係数 $b_{\\mathrm{pref}}$'), (4.56, '向きの係数 $b_{\\mathrm{dir}}$')):
        ax.text(x, 1.0, txt, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7.5)
    ax.set_title('(a) 主の回帰の係数', loc='left', pad=13)
    ax.set_ylabel('最終5点の平均')

    # (b) two-choice BA, preference label and direction label
    ax = axes[1]
    groups = []
    for x0, q in ((0, 'BA_pref'), (3.5, 'BA_dir')):
        es = [sec[f'cpm|FP|{c}'][q] for c in CONDS]
        per = {s: [e['per_pair'][s] for e in es] for s in range(6)}
        groups.append(([x0, x0 + 1, x0 + 2], per, [e['t'] for e in es]))
        vals[f'{q}_pairs'] = {ps.pair_label(s): per[s] for s in range(6)}
    vals['BA'] = draw(ax, groups, XT * 2, 0.5)
    ax.axvline(2.75, color='0.8', lw=0.6, zorder=0)
    ax.set_xlim(-0.4, 6.0)
    for x, txt in ((1.06, '好みのラベル'), (4.56, '向きのラベル')):
        ax.text(x, 1.0, txt, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7.5)
    ax.set_title('(b) 二択の BA', loc='left', pad=13)

    # (c) preference accuracy on the 280 disagreeing episodes
    ax = axes[2]
    es = [sec[f'cpm|FP|{c}']['acc_disagree'] for c in CONDS]
    per = {s: [e['per_pair'][s] for e in es] for s in range(6)}
    vals['acc_disagree'] = draw(ax, [([0, 1, 2], per, [e['t'] for e in es])], XT, 0.5)
    vals['acc_disagree_pairs'] = {ps.pair_label(s): per[s] for s in range(6)}
    ax.set_xlim(-0.4, 2.5)
    ax.text(1.06, 1.0, '食い違う 280 件', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7.5)
    ax.set_title('(c) 好みの正解率', loc='left', pad=13)

    ax.set_ylabel('最終5点の平均')
    lax = fig.add_subplot(gs[1, 2:])
    lax.axis('off')
    h = [plt.Line2D([], [], color=ps.C_MAIN, marker='o', ms=3.0, lw=0.8, label='組1〜3'),
         plt.Line2D([], [], color=ps.C_EXTRA, marker='s', mfc='white', ms=3.0, lw=0.8, label='組4〜6'),
         plt.Line2D([], [], color='k', marker='s', ms=3.6, lw=1.5, label='6組の平均と95%信頼区間（$t$，自由度5）'),
         plt.Line2D([], [], color='0.6', lw=0.6, ls='--', label='0.5（偶然の水準）')]
    lax.legend(handles=h, loc='center left', frameon=False)
    fig.tight_layout(w_pad=1.2, h_pad=1.2)
    ps.save(fig, 'v3_discrim')
    json.dump(vals, open(os.path.join(ps.FIGS, 'v3_discrim_values.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
