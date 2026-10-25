"""
Paper figure 6 'inverse' (experiment 4 = v14-A; documents only, CPU).
(a) BA vs T on the pref test with the self-vision error (L1), l1-stage late-5 mean with the range as bars, one panel per
    pair 1..6: 80% (v13), 50% (v12), no experience (v12 random), no sharing (v13-NS, pairs 1-3), plus the direction-only
    baseline (paper/check/direction_baseline.json, mean and 95% CI band) - same source files as
    analyze/v14/fig_summary_v14.fig1.
(b) share following preference in the 88 episodes where preference and A-2's direction disagree, late-5 mean and range:
    other-side feature-prediction error (FP) on the mse-stage late-5, self-vision L1 on the l1-stage late-5
    (paper/check/follow_pref_late5.json); the number of points with CI low > 0.5 is written for 80%.
Writes paper/figs/inverse.{pdf,png}.
"""

import json
import os

import numpy as np

from paper_style import (C_50, C_80, C_NONE, C_NS, M_50, M_80, M_NONE, M_NS, PI, RES, pair_label, plt, save, setup)

TS = (5, 10, 20, 30, 50, 100)
EP = ('160', '170', '180', '190', '200')
J = lambda *p: json.load(open(os.path.join(*p)))  # noqa


def curve(detail, pair, key):
    a = np.array([[detail[pair][f'pref_L1_T{T}']['points'][e][key] for e in EP] for T in TS])
    return a.mean(1), a.min(1), a.max(1)


def main():
    setup()
    inv = os.path.join(RES, 'v14', 'inverse')
    v13 = J(inv, 'aggregate_v13_s0to5.json')['detail']
    v12 = J(inv, 'aggregate_v12_s0to5.json')['detail']
    ns = J(inv, 'aggregate_v13_vs_ns.json')['detail']
    db = J(PI, 'paper', 'check', 'direction_baseline.json')
    fp = J(PI, 'paper', 'check', 'follow_pref_late5.json')['summary']
    fig = plt.figure(figsize=(6.3, 7.4))
    outer = fig.add_gridspec(2, 1, height_ratios=(2.1, 1.0), hspace=0.36)
    gs = outer[0].subgridspec(2, 3, hspace=0.42, wspace=0.12)
    x = np.arange(len(TS))
    dm = np.array([db[f'T{T}|all']['ba'] for T in TS])
    dlo = np.array([db[f'T{T}|all']['ba_ci'][0] for T in TS])
    dhi = np.array([db[f'T{T}|all']['ba_ci'][1] for T in TS])
    axa = []
    for s in range(6):
        ax = fig.add_subplot(gs[s // 3, s % 3], sharey=axa[0] if axa else None)
        axa.append(ax)
        ax.fill_between(x, dlo, dhi, color='0.85', lw=0, zorder=0)
        ax.plot(x, dm, color='k', ls='--', lw=1.0, zorder=1, label='方角だけの基準（95%信頼区間）')
        rows = [(C_80, M_80, '経験80%', v13, f'v13:mixed:{s},v12:random:{s}', 'ba_exp'),
                (C_50, M_50, '経験50%', v12, f'v12:mixed:{s},v12:random:{s}', 'ba_exp'),
                (C_NONE, M_NONE, '経験なし', v12, f'v12:mixed:{s},v12:random:{s}', 'ba_noexp')]
        if s < 3:
            rows.append((C_NS, M_NS, '非共有（組1〜3）', ns, f'v13:mixed:{s},v13ns:mixed:{s}', 'ba_noexp'))
        for j, (c, mk, lab, d, pr, key) in enumerate(rows):
            m, lo, hi = curve(d, pr, key)
            xx = x + (j - 1.5) * 0.07
            ax.errorbar(xx, m, yerr=[m - lo, hi - m], color=c, marker=mk, ms=3, lw=1.0, capsize=1.5, label=lab,
                        mfc='white' if mk == M_NS else c)
        ax.axhline(0.5, color='0.3', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([str(T) for T in TS])
        ax.set_title(pair_label(s) + ('' if s < 3 else '（予備）' if s == 3 else '（追加）'), fontsize=8.5)
        if s % 3 == 0:
            ax.set_ylabel('正解率（BA）')
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if s >= 3:
            ax.set_xlabel('照合の長さ $T$（フレーム）')
    axa[0].set_ylim(0.3, 0.85)
    axa[0].text(-0.33, 1.22, '(a) 自己視界基準の正解率と照合の長さ', transform=axa[0].transAxes, fontsize=9)
    h, l = axa[0].get_legend_handles_labels()
    order = [1, 2, 3, 4, 0]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc='upper center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.8, columnspacing=1.0)
    # (b)
    gsb = outer[1].subgridspec(1, 2, wspace=0.08)
    conds = [('80', C_80, M_80, 'v13{}_mixed_{}', range(6)), ('50', C_50, M_50, 'v12{}_mixed_{}', range(6)),
             ('none', C_NONE, M_NONE, 'v12{}_random_{}', range(6)), ('ns', C_NS, M_NS, 'v13ns{}_mixed_{}', range(3))]
    axb = []
    for k, (q, stage, title) in enumerate((('FP', 'mse', '他者特徴基準（MSE 期間の最終5点）'),
                                          ('L1', '', '自己視界基準（L1 期間の最終5点）'))):
        ax = fig.add_subplot(gsb[0, k], sharey=axb[0] if axb else None)
        axb.append(ax)
        for j, (cn, c, mk, pat, seeds) in enumerate(conds):
            for s in seeds:
                key = pat.format(stage, s) + f'|{q}'
                if key not in fp:          # 50%, pair 4, mse stage: not saved
                    continue
                r = fp[key]
                xx = s + (j - 1.5) * 0.17
                m = r['late5_mean']
                ax.errorbar(xx, m, yerr=[[m - r['late5_min']], [r['late5_max'] - m]], color=c, marker=mk, ms=3.5,
                            lw=1.0, capsize=1.5, mfc='white' if mk == M_NS else c)
                if cn == '80':
                    ax.text(xx, r['late5_max'] + 0.03, f"{r['n_ci_low_above_half']}/5", ha='center', fontsize=6.5,
                            color=C_80)
        ax.axhline(0.5, color='0.3', lw=0.6, ls='--')
        ax.set_xticks(range(6))
        ax.set_xticklabels([pair_label(s) for s in range(6)])
        ax.set_title(title, fontsize=8.5)
        if k == 0:
            ax.set_ylabel('好み追従率')
            ax.text(-0.16, 1.2, '(b) 好みと方角が食い違う88エピソードで選択が好みに従う割合', transform=ax.transAxes,
                    fontsize=9)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    axb[0].set_ylim(0.15, 0.85)
    save(fig, 'inverse')


if __name__ == '__main__':
    main()
