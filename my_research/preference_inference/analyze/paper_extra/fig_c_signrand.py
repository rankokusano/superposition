"""
Paper extra C figure (docs/paper_extra_log.md Sec.25; central-figure quantities). Conditions on the x axis
(no experience, sign-randomised, zero pleasure if its evaluation exists, 80%), one point per pair (pairs 1-3, markers
o / s / ^), late-5 mean with the range of the 5 points as a bar:
  (a) G3 hold g (mse late-5), (b)(c) sign-following part green / mixed (l1 late-5), (d) G5b (l1 late-5),
  (e) d_P = (g_hold - g_flip)/2 (mse late-5), (f) true-pleasure Delta (l1 ep200; filled green-liking, hollow mixed).
From C_signrand/summary.json only (no new computation).
-> data/result/paper_extra/C_signrand/figs/c_signrand.{pdf,png}
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'paper'))
import paper_style as ps  # noqa

C = os.path.join(_PI, 'data', 'result', 'paper_extra', 'C_signrand')
ps.FIGS = os.path.join(C, 'figs')
C_SR, C_PZ = '#8a3ffc', '#b8860b'      # same colours as c_curves
PAIR_MK = ('o', 's', '^')
BREAK = {'経験なし': '経験\nなし', '符号無作為': '符号\n無作為', '快・不快 零': '快・不快\n零', '経験80%': '経験\n80%'}


def main():
    ps.setup()
    plt = ps.plt
    R = json.load(open(os.path.join(C, 'summary.json')))
    conds = [('random', 'signrand', ps.COND_LABEL['none'], ps.C_NONE), ('signrand', 'signrand', '符号無作為', C_SR)]
    if 'pzero' in R:
        conds.append(('pzero', 'pzero', '快・不快 零', C_PZ))
    conds.append(('mixed80', 'signrand', ps.COND_LABEL['80'], ps.C_80))
    panels = (('G3_hold', '(a) 自己側の積み上がり g（P を保つ）'), ('signfollow_green', '(b) 符号に従う分（緑好み）'),
              ('signfollow_pref', '(c) 符号に従う分（好み混合）'), ('G5b', '(d) 他者側の符号応答 G5b'),
              ('G3_dP', '(e) P への依存 $d_P$'), ('delta', '(f) 真値条件 Δ（真値 − 零）'))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for ax, (q, title) in zip(axes.flat, panels):
        for i, (c, src, _, col) in enumerate(conds):
            P = R[src]['per_pair']
            for s in range(3):
                if f's{s}' not in P:
                    continue
                d = P[f's{s}']['values'][c]
                x = i + (s - 1) * 0.2
                if q == 'delta':
                    for k, mfc, dx in (('green', col, -0.04), ('pref', 'white', 0.04)):
                        ax.plot(x + dx, d[f'delta_{k}'], PAIR_MK[s], color=col, mfc=mfc, ms=3.6)
                    continue
                v = np.asarray(d[q], float)
                ax.errorbar(x, v.mean(), yerr=[[v.mean() - v.min()], [v.max() - v.mean()]], fmt=PAIR_MK[s], color=col,
                            ms=3.6, lw=0.7, capsize=1.2)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([BREAK.get(lab, lab) for _, _, lab, _ in conds], fontsize=7)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_title(title, fontsize=8)
    for s in range(3):
        axes[0, 0].plot([], [], PAIR_MK[s], color='0.3', ms=3.6, label=ps.pair_label(s))
    axes[0, 0].legend(frameon=False, fontsize=6.5, loc='upper left')
    fig.text(0.99, -0.04, '点：最終5点の平均（棒は5点の範囲）。(f) は L1 期間 ep200 の1点、塗り：緑好みデータ、白抜き：好み混合データ。'
             '$d_P$ =（保つ − 反転）/ 2', ha='right', fontsize=6.5)
    fig.tight_layout()
    ps.save(fig, 'c_signrand')


if __name__ == '__main__':
    main()
