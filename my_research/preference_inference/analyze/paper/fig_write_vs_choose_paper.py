"""
Paper figure 5 'write_vs_choose' (documents only, CPU): among the 88 episodes where A-2's preference and its direction
over t = 1..50 disagree, the share of choices (experiment 4 = v14-A) or of the one-bit module's u (v14-B) that follows
the preference; 50% and 80%, pairs 1-3, late-5 mean with the range as bars.
  choose, self-vision L1: l1-stage late-5;  choose, FP: mse-stage late-5  (paper/check/follow_pref_late5.json)
  write out, one-bit u: stage-2 late-5     (data/result/v14/stage2u/u_direction{,_v13}.json, '<tag>|ep<e>|all')
Same comparison as data/result/v14/summary_figs/fig3_write_vs_choose.png (lower row), there at ep200 with CIs.
Writes paper/figs/write_vs_choose.{pdf,png} and paper/figs/write_vs_choose_values.json.
"""

import json
import os

import numpy as np

from paper_style import FIGS, PI, RES, pair_label, plt, save, setup

EP = (160, 170, 180, 190, 200)
METH = (('当てて選ぶ：自己視界基準（L1 期間）', '#2a78d6', 'o'), ('当てて選ぶ：他者特徴基準（MSE 期間）', '#1baf7a', 's'),
        ('書き出す：1ビットモジュール（段階II）', '#eb6834', 'D'))


def main():
    setup()
    fp = json.load(open(os.path.join(PI, 'paper', 'check', 'follow_pref_late5.json')))['summary']
    ud = {'50': json.load(open(os.path.join(RES, 'v14', 'stage2u', 'u_direction.json'))),
          '80': json.load(open(os.path.join(RES, 'v14', 'stage2u', 'u_direction_v13.json')))}
    utag = {'50': 'mixed_s{}', '80': 'mixed80_s{}'}
    ctag = {'50': ('v12_mixed_{}|L1', 'v12mse_mixed_{}|FP'), '80': ('v13_mixed_{}|L1', 'v13mse_mixed_{}|FP')}
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    vals = {}
    for ax, (cn, title) in zip(axs, (('50', '(a) 経験50%条件'), ('80', '(b) 経験80%条件'))):
        for s in range(3):
            u = [ud[cn][f'{utag[cn].format(s)}|ep{e}|all']['disagree_follow_pref'] for e in EP]
            rows = [(fp[ctag[cn][0].format(s)]['late5_min'], fp[ctag[cn][0].format(s)]['late5_mean'], fp[ctag[cn][0].format(s)]['late5_max']),
                    (fp[ctag[cn][1].format(s)]['late5_min'], fp[ctag[cn][1].format(s)]['late5_mean'], fp[ctag[cn][1].format(s)]['late5_max']),
                    (min(u), float(np.mean(u)), max(u))]
            vals[f'{cn}|{pair_label(s)}'] = {m[0]: r for m, r in zip(METH, rows)}
            for j, ((lo, m, hi), (lab, c, mk)) in enumerate(zip(rows, METH)):
                open_mk = j == 2 and cn == '50'   # one-bit following share: pre-registered only for 80% (v14 log Sec.29)
                if j == 2:
                    lab = lab + ('（事前登録外）' if open_mk else '（事前登録）')
                ax.errorbar(s + (j - 1) * 0.22, m, yerr=[[m - lo], [hi - m]], color=c, marker=mk, ms=4.5 if open_mk else 4,
                            lw=1.0, capsize=2, mfc='white' if open_mk else c, mew=1.2,
                            ls='none', label=lab if s == 0 else None)
        ax.axhline(0.5, color='0.3', lw=0.6, ls='--')
        ax.set_xticks(range(3))
        ax.set_xticklabels([pair_label(s) for s in range(3)])
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(title, fontsize=9)
    axs[0].set_ylabel('好み追従率')
    axs[0].set_ylim(0.2, 0.8)
    h, l = axs[0].get_legend_handles_labels()
    h2, l2 = axs[1].get_legend_handles_labels()
    fig.legend(h + h2[2:], l + l2[2:], loc='lower center', ncol=2, frameon=False, fontsize=7, columnspacing=1.0,
               handlelength=1.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, 'write_vs_choose')
    json.dump(vals, open(os.path.join(FIGS, 'write_vs_choose_values.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
