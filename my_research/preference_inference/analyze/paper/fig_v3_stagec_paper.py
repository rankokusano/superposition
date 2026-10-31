"""
Paper v3 figure for experiment 3, stage C (paper/draft_v3, 2026-11-01). No new computation: the saved
pre-registered aggregation only (data/result/paper_extra/StageC/analysis/summary.json; stageC_analysis.py c1a0a17,
pre-registration 3ade5ed, amendment 1de4a37, results 74f33bb, paper/check/stageC_G.md Sec.2, 4, 5, 13.3).
  (a) main judgement: b^q_pref of stage 1 (standardised q~_e, all 1200 episodes) for no experience / 50% / before
      training of G (G at 00000.pth on the 50% stage-I predictor);
  (b) basic read-out BA of the preference (1200 episodes, nested-CV C) from the observation (O), the state of G (S),
      the output Q2 (Q) and h^2 (H), for no experience / 50% / before training;
  (c) preference accuracy on the 280 disagreeing episodes of the read-out fitted on the agreeing episodes, before
      training and after training (50%).
Pairs 1-3 grey circles, pair 4 orange square, one thin line per pair; black square with bar = mean of the 4 pairs and
t 95% CI (df 3), checked against the saved t CI. (O) is the same for every model, so its t CI has width 0.
-> paper/figs/v3_stagec.{pdf,png}, v3_stagec_values.json
"""

import json
import os

import paper_style as ps
from fig_v3_paper import TEXTWIDTH_IN, mean_ci

SUMMARY = os.path.join(ps.PI, 'data', 'result', 'paper_extra', 'StageC', 'analysis', 'summary.json')
REPS = (('O', '観察'), ('S', '$G$ の内部状態'), ('Q', '出力 $Q2$'), ('H', '$h^2$'))
TICK = {'none': '経験\nなし', 'p50': '経験\n50%', 'pre': '学習\n前', 'pre_t': '学習\n前', 'p50_t': '学習\n済み'}


def draw(ax, groups, ref):
    """groups: list of (x positions, {pair: [values]}, saved t summaries)."""
    out = {}
    for xs, per, saved in groups:
        for s, ys in per.items():
            col = ps.C_MAIN if s <= 2 else ps.C_EXTRA
            ax.plot(xs, ys, '-', marker='o' if s <= 2 else 's', color=col, ms=3.0, lw=0.8,
                    mfc=col if s <= 2 else 'white', mew=0.8, zorder=2)
        for i, x in enumerate(xs):
            m, lo, hi = mean_ci([per[s][i] for s in per])
            t = saved[i]
            assert abs(m - t['mean']) < 1e-9 and abs(lo - t['ci'][0]) < 1e-9 and abs(hi - t['ci'][1]) < 1e-9, (x, m, t)
            ax.errorbar(x + 0.15, m, yerr=[[m - lo], [hi - m]], fmt='s', color='k', ms=3.6, lw=1.5, capsize=2.5, zorder=3)
            out[str(x)] = [m, lo, hi]
    ax.axhline(ref, color='0.6', lw=0.6, ls='-' if ref == 0 else '--', zorder=0)
    ax.tick_params(axis='x', labelsize=7)
    return out


def group_label(ax, x, txt):
    ax.text(x, 1.0, txt, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7.5)


def main():
    ps.setup()
    plt = ps.plt
    S = json.load(open(SUMMARY))
    L1, R = S['L1'], S['readout']
    vals = {}

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 5.0))
    gs = fig.add_gridspec(2, 14)
    ax_a = fig.add_subplot(gs[0, :4])
    ax_b = fig.add_subplot(gs[0, 4:])
    ax_c = fig.add_subplot(gs[1, :9])
    lax = fig.add_subplot(gs[1, 9:])

    # (a) stage 1 b^q_pref (standardised), 95% t CI
    conds = ('none', 'p50', 'pre')
    es = [L1[f'{c}|s1|std|pref']['95']['t'] for c in conds]
    per = {s: [e['per_pair'][s] for e in es] for s in range(4)}
    vals['a_bq_pref_stage1'] = draw(ax_a, [([0, 1, 2], per, es)], 0)
    vals['a_pairs'] = {ps.pair_label(s): per[s] for s in range(4)}
    ax_a.set_xticks([0, 1, 2])
    ax_a.set_xticklabels([TICK[c] for c in conds])
    ax_a.set_xlim(-0.5, 2.6)
    ax_a.set_title('(a) 主の判定', loc='left', pad=13)
    group_label(ax_a, 1.07, '第1段の $b^{q}_{\\mathrm{pref}}$')
    ax_a.set_ylabel('最終5点の平均')

    # (b) basic read-out BA
    groups, ticks, xs_all = [], [], []
    x0 = 0.0
    centers = []
    for rep, name in REPS:
        cs = ('p50',) if rep == 'O' else conds
        xs = [x0 + i for i in range(len(cs))]
        es = [R[f'nest|{rep}|{c}|BA']['t'] for c in cs]
        per = {s: [e['per_pair'][s] for e in es] for s in range(4)}
        groups.append((xs, per, es))
        ticks += ['共通'] if rep == 'O' else [TICK[c] for c in cs]
        xs_all += xs
        centers.append((sum(xs) / len(xs) + 0.07, name))
        vals[f'b_BA_{rep}_pairs'] = {ps.pair_label(s): per[s] for s in range(4)}
        x0 = xs[-1] + 1.6
    vals['b_BA'] = draw(ax_b, groups, 0.5)
    ax_b.set_xticks(xs_all)
    ax_b.set_xticklabels(ticks)
    for xs, _, _ in groups[1:]:
        ax_b.axvline(xs[0] - 0.8, color='0.8', lw=0.6, zorder=0)
    for x, name in centers:
        group_label(ax_b, x, name)
    ax_b.set_xlim(-0.6, xs_all[-1] + 0.6)
    ax_b.set_title('(b) 好みの基本の読み出しの BA', loc='left', pad=13)

    # (c) generalisation: agreeing -> 280 disagreeing, before / after training
    groups, ticks, xs_all, centers = [], [], [], []
    x0 = 0.0
    for rep, name in REPS:
        cs = ('p50',) if rep == 'O' else ('pre', 'p50')
        xs = [x0 + i for i in range(len(cs))]
        es = [R[f'nest|{rep}|{c}|gen_dis']['t'] for c in cs]
        per = {s: [e['per_pair'][s] for e in es] for s in range(4)}
        groups.append((xs, per, es))
        ticks += ['共通'] if rep == 'O' else [TICK['pre_t'], TICK['p50_t']]
        xs_all += xs
        centers.append((sum(xs) / len(xs) + 0.07, name))
        vals[f'c_gen_dis_{rep}_pairs'] = {ps.pair_label(s): per[s] for s in range(4)}
        x0 = xs[-1] + 1.6
    vals['c_gen_dis'] = draw(ax_c, groups, 0.5)
    ax_c.set_xticks(xs_all)
    ax_c.set_xticklabels(ticks)
    for xs, _, _ in groups[1:]:
        ax_c.axvline(xs[0] - 0.8, color='0.8', lw=0.6, zorder=0)
    for x, name in centers:
        group_label(ax_c, x, name)
    ax_c.set_xlim(-0.6, xs_all[-1] + 0.6)
    ax_c.set_title('(c) 一致する場面で当てはめ，食い違う 280 件で採点', loc='left', pad=13)
    ax_c.set_ylabel('好みに当たる割合')

    lax.axis('off')
    h = [plt.Line2D([], [], color=ps.C_MAIN, marker='o', ms=3.0, lw=0.8, label='組1〜3'),
         plt.Line2D([], [], color=ps.C_EXTRA, marker='s', mfc='white', ms=3.0, lw=0.8, label='組4'),
         plt.Line2D([], [], color='k', marker='s', ms=3.6, lw=1.5, label='4組の平均と95%信頼区間\n（$t$，自由度3）'),
         plt.Line2D([], [], color='0.6', lw=0.6, ls='--', label='0.5（偶然の水準）')]
    lax.legend(handles=h, loc='center left', frameon=False)
    fig.tight_layout(w_pad=0.8, h_pad=1.2)
    ps.save(fig, 'v3_stagec')
    json.dump(vals, open(os.path.join(ps.FIGS, 'v3_stagec_values.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
