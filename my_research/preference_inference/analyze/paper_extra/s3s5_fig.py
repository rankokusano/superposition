"""
Paper round 6 (docs/paper_extra_log.md Sec.34.8; instructions Sec.5.7): per-pair differences signrand - 80% (late-5
mean of each model, the main form) for pairs 1-6, with the across-pair mean and t 95% CI (df 5), from
C_signrand/s3s5/summary.json. Common test (d_P, G5b, sign-following 1/2) and B4 data (G5b, sign-following 1/2).
Filled marker = the pair passed its per-pair judgement vs 80% (3 of 5 points), open = it did not.
Style from analyze/paper/paper_style.py (Japanese font, Type 42), but written to the working location, not paper/figs:
-> data/result/paper_extra/C_signrand/s3s5/fig_signrand_6pairs.{pdf,png}
Needs the font copied into the container first (see paper_style.py).
"""

import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'paper'))
import paper_style as PS  # noqa
plt = PS.plt

D = os.path.join(_PI, 'data', 'result', 'paper_extra', 'C_signrand', 's3s5')
S = json.load(open(os.path.join(D, 'summary.json')))
PANELS = [('共通の評価データ', [('dP', '$d_P$\n（主）'), ('G5b', 'G5b\n（主）'),
                              ('signfollow_green', '符号に従う分 ①\n（副）'), ('signfollow_pref', '符号に従う分 ②\n（副）')]),
          ('追跡のできる評価データ（B4）', [('B4_G5b', 'G5b\n（副）'), ('B4_signfollow_green', '符号に従う分 ①\n（副）'),
                                     ('B4_signfollow_pref', '符号に従う分 ②\n（副）')])]
DX = np.array([-0.27, -0.20, -0.13, -0.06, 0.01, 0.08])   # pairs 1-6 side by side; mean and CI at +0.22


def save(fig, name):
    fig.savefig(os.path.join(D, f'{name}.png'), dpi=200)
    full = os.path.join(D, f'{name}_fullfont.pdf')
    fig.savefig(full)
    subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.5',
                    '-dSubsetFonts=true', '-dEmbedAllFonts=true', '-dAutoRotatePages=/None',
                    f'-sOutputFile={os.path.join(D, name + ".pdf")}', full], check=True)
    os.remove(full)
    print('wrote', os.path.join(D, name + '.{pdf,png}'))


def main():
    PS.setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, gridspec_kw={'width_ratios': [4, 3]})
    for ax, (title, qs) in zip(axes, PANELS):
        ax.axhline(0, color='0.6', lw=0.6, zorder=0)
        for i, (q, lab) in enumerate(qs):
            A = S['across6'][q]
            for s in range(6):
                ok = S['per_pair'][f's{s}'][f'{q}_vs80']['pair_ok']
                c, m = (PS.C_MAIN, 'o') if s < 3 else (PS.C_EXTRA, 's')
                ax.plot(i + DX[s], A['diff_means'][s], m, ms=4.5, mec=c, mew=0.9, mfc=c if ok else 'white', zorder=3)
                if not ok:
                    ax.annotate(PS.pair_label(s), (i + DX[s], A['diff_means'][s]), xytext=(-3, 5),
                                textcoords='offset points', ha='right', fontsize=6.5, color='0.25')
            t = A['all6_means']
            ax.errorbar(i + 0.22, t['mean'], yerr=[[t['mean'] - t['ci95'][0]], [t['ci95'][1] - t['mean']]],
                        fmt='D', color='k', ms=4, capsize=2.5, lw=1.0, zorder=4)
        ax.set_xticks(range(len(qs)))
        ax.set_xticklabels([lab for _, lab in qs])
        ax.set_xlim(-0.5, len(qs) - 0.5)
        ax.set_title(title)
    axes[0].set_ylabel('符号無作為 − 経験80%\n（最終5点の平均の差）')
    h = [plt.Line2D([], [], ls='', marker='o', ms=4.5, mec=PS.C_MAIN, mfc=PS.C_MAIN, label='組1〜3'),
         plt.Line2D([], [], ls='', marker='s', ms=4.5, mec=PS.C_EXTRA, mfc=PS.C_EXTRA, label='組4〜6'),
         plt.Line2D([], [], ls='', marker='s', ms=4.5, mec='0.3', mfc='white', label='組ごとの判定で不合格'),
         plt.Line2D([], [], ls='-', marker='D', ms=4, color='k', label='6組の平均と 95% CI（$t$，自由度5）')]
    fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'fig_signrand_6pairs')


if __name__ == '__main__':
    main()
