"""
Paper v3 figures (paper/draft_v3, 2026-10-31). No new computation: saved summaries only.
  v3_self : (a) amount of experience and g (G3, MSE late-5), pairs 1-6, as panel (a) of the central figure
            (data/result/v13/extra/across_pairs.json); (b) d_P for the 50% and 80% conditions, normal vs sign-randomised:
            50% from paper_extra/E1_signrand50/summary.json ('50', 'sr50'; log Sec.35.11),
            80% from paper_extra/C_signrand/s3s5/summary.json ('mixed80', 'signrand'; log Sec.34.8).
  v3_other: G5b and the sign-following part (green, mixed) for 50% and 80%, each with no experience / normal /
            sign-randomised (same two summaries; the no-experience models are the same in both and are asserted equal).
Pairs 1-3 grey circles, pairs 4-6 orange squares, one thin line per pair; black square with bar = mean of the 6 pairs
and t 95% CI (df 5) of the per-pair values. Each per-pair value = mean of the 5 late checkpoints.
-> paper/figs/v3_self.{pdf,png}, v3_other.{pdf,png}, v3_values.json
"""

import json
import os

import numpy as np
from scipy import stats

import paper_style as ps

R = os.path.join(ps.PI, 'data', 'result')
E1 = os.path.join(R, 'paper_extra', 'E1_signrand50', 'summary.json')
S6 = os.path.join(R, 'paper_extra', 'C_signrand', 's3s5', 'summary.json')
AP = os.path.join(R, 'v13', 'extra', 'across_pairs.json')
TEXTWIDTH_IN = 406.86 / 72.27   # drawn at the text width (jsarticle 11pt A4) so that the smallest text is 7 pt
OTHER = (('G5b', '(a) 他者側の符号応答 G5b'), ('signfollow_green', '(b) 符号に従う分（緑好み）'),
         ('signfollow_pref', '(c) 符号に従う分（好み混合）'))


def late5(v):
    return float(np.mean(np.asarray(v, float)))


def load():
    e1 = json.load(open(E1))['values']
    s6 = json.load(open(S6))['values']
    V = {}
    for s in range(6):
        k = f's{s}'
        for q in ('dP', 'G5b', 'signfollow_green', 'signfollow_pref'):   # the no-experience models are shared
            assert np.array_equal(np.asarray(e1[k]['none'][q], float), np.asarray(s6[k]['random'][q], float)), (k, q)
        V[s] = {'none': s6[k]['random'], '50': e1[k]['50'], 'sr50': e1[k]['sr50'],
                '80': s6[k]['mixed80'], 'sr80': s6[k]['signrand']}
    return V


def mean_ci(vals):
    a = np.asarray(vals, float)
    h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean()), float(a.mean() - h), float(a.mean() + h)


def draw_groups(ax, groups, xt):
    """groups: list of (x positions, {pair: [values]}). Thin line per pair, mean and CI per position."""
    out = {}
    for xs, per in groups:
        for s, ys in per.items():
            col = ps.C_MAIN if s <= 2 else ps.C_EXTRA
            mk = 'o' if s <= 2 else 's'
            ax.plot(xs, ys, '-', marker=mk, color=col, ms=3.0, lw=0.8, mfc=col if s <= 2 else 'white', mew=0.8,
                    zorder=2)
        for i, x in enumerate(xs):
            m, lo, hi = mean_ci([per[s][i] for s in per])
            ax.errorbar(x + 0.12, m, yerr=[[m - lo], [hi - m]], fmt='s', color='k', ms=3.6, lw=1.5, capsize=2.5,
                        zorder=3)
            out[str(x)] = [m, lo, hi]
    ax.axhline(0, color='0.6', lw=0.5, zorder=0)
    ax.set_xticks([x for xs, _ in groups for x in xs])
    ax.set_xticklabels(xt)
    return out


def legend(fig):
    h = [ps.plt.Line2D([], [], color=ps.C_MAIN, marker='o', ms=3.0, lw=0.8, label='組1〜3'),
         ps.plt.Line2D([], [], color=ps.C_EXTRA, marker='s', mfc='white', ms=3.0, lw=0.8, label='組4〜6'),
         ps.plt.Line2D([], [], color='k', marker='s', ms=3.6, lw=1.5, label='6組の平均と95%信頼区間（$t$，自由度5）')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False)


def main():
    ps.setup()
    plt = ps.plt
    V = load()
    ap = json.load(open(AP))['per_pair']
    vals = {'self': {}, 'other': {}}

    # ---------------- v3_self
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.8), gridspec_kw={'width_ratios': [3, 4]})
    ax = axes[0]
    per = {int(k[1:]): [ap[k]['G3'][c] for c in ('none', '50', '80')] for k in ap}
    for s in per:     # the same g as the summaries used for (b) (late-5 means of G3 hold)
        assert abs(per[s][0] - late5(V[s]['none']['G3_hold'])) < 1e-9 and abs(per[s][2] - late5(V[s]['80']['G3_hold'])) < 1e-9 \
            and abs(per[s][1] - late5(V[s]['50']['G3_hold'])) < 1e-9, s
    vals['self']['g'] = draw_groups(ax, [([0, 1, 2], per)], ['経験なし', '経験50%', '経験80%'])
    vals['self']['g_pairs'] = {ps.pair_label(s): v for s, v in per.items()}
    ax.set_xlim(-0.4, 2.5)
    ax.set_title('(a) 経験の量と自分側の積み上がり $g$', loc='left')
    ax.set_ylabel('最終5点の平均')
    ax = axes[1]
    p50 = {s: [late5(V[s]['50']['dP']), late5(V[s]['sr50']['dP'])] for s in range(6)}
    p80 = {s: [late5(V[s]['80']['dP']), late5(V[s]['sr80']['dP'])] for s in range(6)}
    vals['self']['dP'] = draw_groups(ax, [([0, 1], p50), ([2.5, 3.5], p80)],
                                     ['経験50%', '符号無作為\n（50%）', '経験80%', '符号無作為\n（80%）'])
    vals['self']['dP_pairs'] = {ps.pair_label(s): {'50': p50[s], '80': p80[s]} for s in range(6)}
    ax.axvline(1.75, color='0.8', lw=0.6, zorder=0)
    ax.set_xlim(-0.4, 4.0)
    ax.set_title('(b) 自分側の快・不快への依存 $d_P$', loc='left')
    legend(fig)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.5)
    ps.save(fig, 'v3_self')

    # ---------------- v3_other
    # rows: the 50% and the 80% conditions (each with no experience / normal / sign-randomised); columns: quantities.
    # Six positions in one panel do not fit at the text width with 7 pt labels, so the two amounts are two rows.
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH_IN, 4.6), sharey='col')
    for j, (q, title) in enumerate(OTHER):
        for i, (amt, lab) in enumerate((('50', '経験50%'), ('80', '経験80%'))):
            ax = axes[i, j]
            per = {s: [late5(V[s][c][q]) for c in ('none', amt, 'sr' + amt)] for s in range(6)}
            vals['other'][f'{q}_{amt}'] = draw_groups(ax, [([0, 1, 2], per)], ['経験\nなし', '経験\n' + amt + '%', '符号\n無作為'])
            vals['other'][f'{q}_{amt}_pairs'] = {ps.pair_label(s): per[s] for s in range(6)}
            ax.set_xlim(-0.4, 2.5)
            ax.tick_params(axis='x', labelsize=7)
            if i == 0:
                ax.set_title(title, loc='left', fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f'{lab}の組\n最終5点の平均')
    legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.0, h_pad=1.0)
    ps.save(fig, 'v3_other')
    json.dump(vals, open(os.path.join(ps.FIGS, 'v3_values.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
