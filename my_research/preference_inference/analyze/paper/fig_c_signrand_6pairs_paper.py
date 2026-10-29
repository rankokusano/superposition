"""
Supplementary figure S1 (S4) for the paper (paper round 9): per-pair differences signrand - 80% for pairs 1-6, with the
across-pair mean and t 95% CI (df 5). A copy of analyze/paper_extra/s3s5_fig.py (kept unchanged) with only the axis
labels changed: the circled numbers for the sign-following part are replaced by the data names (緑好み / 好み混合),
which keeps the labels inside JIS X 0208. Values, markers, colours and layout are as in s3s5_fig.py.
From C_signrand/s3s5/summary.json only (no new computation). -> paper/figs/c_signrand_6pairs.{pdf,png}
"""

import json
import os

import numpy as np

import paper_style as ps

D = os.path.join(ps.PI, 'data', 'result', 'paper_extra', 'C_signrand', 's3s5')
PANELS = [('共通の評価データ', [('dP', '$d_P$\n（主）'), ('G5b', 'G5b\n（主）'),
                              ('signfollow_green', '符号に従う分\n（緑好み，副）'),
                              ('signfollow_pref', '符号に従う分\n（好み混合，副）')]),
          ('追跡のできる評価データ（B4）', [('B4_G5b', 'G5b\n（副）'),
                                     ('B4_signfollow_green', '符号に従う分\n（緑好み，副）'),
                                     ('B4_signfollow_pref', '符号に従う分\n（好み混合，副）')])]
DX = np.array([-0.27, -0.20, -0.13, -0.06, 0.01, 0.08])   # pairs 1-6 side by side; mean and CI at +0.22


def main():
    ps.setup()
    plt = ps.plt
    S = json.load(open(os.path.join(D, 'summary.json')))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, gridspec_kw={'width_ratios': [4, 3]})
    for ax, (title, qs) in zip(axes, PANELS):
        ax.axhline(0, color='0.6', lw=0.6, zorder=0)
        for i, (q, lab) in enumerate(qs):
            A = S['across6'][q]
            for s in range(6):
                ok = S['per_pair'][f's{s}'][f'{q}_vs80']['pair_ok']
                c, m = (ps.C_MAIN, 'o') if s < 3 else (ps.C_EXTRA, 's')
                ax.plot(i + DX[s], A['diff_means'][s], m, ms=4.5, mec=c, mew=0.9, mfc=c if ok else 'white', zorder=3)
                if not ok:
                    ax.annotate(ps.pair_label(s), (i + DX[s], A['diff_means'][s]), xytext=(-3, 5),
                                textcoords='offset points', ha='right', fontsize=6.5, color='0.25')
            t = A['all6_means']
            ax.errorbar(i + 0.22, t['mean'], yerr=[[t['mean'] - t['ci95'][0]], [t['ci95'][1] - t['mean']]],
                        fmt='D', color='k', ms=4, capsize=2.5, lw=1.0, zorder=4)
        ax.set_xticks(range(len(qs)))
        ax.set_xticklabels([lab for _, lab in qs])
        ax.set_xlim(-0.5, len(qs) - 0.5)
        ax.set_title(title)
    axes[0].set_ylabel('符号無作為 − 経験80%\n（最終5点の平均の差）')
    h = [plt.Line2D([], [], ls='', marker='o', ms=4.5, mec=ps.C_MAIN, mfc=ps.C_MAIN, label='組1〜3'),
         plt.Line2D([], [], ls='', marker='s', ms=4.5, mec=ps.C_EXTRA, mfc=ps.C_EXTRA, label='組4〜6'),
         plt.Line2D([], [], ls='', marker='s', ms=4.5, mec='0.3', mfc='white', label='組ごとの判定で不合格'),
         plt.Line2D([], [], ls='-', marker='D', ms=4, color='k', label='6組の平均と 95% CI（$t$，自由度5）')]
    fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    ps.save(fig, 'c_signrand_6pairs')


if __name__ == '__main__':
    main()
