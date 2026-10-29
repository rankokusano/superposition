"""
Paper round 11, E2 (docs/paper_extra_log.md Sec.35.7): per-pair late-5 means on the main cell (preference-mixed data,
other-side feature-prediction error FP, frames 1-50) from E2/summary.json (e2_aggregate.py) only.
  left  : U_pref = L(opp) - L(true) per condition (pairs 1-6, the 6-pair mean and t 95% CI, df 5)
  middle: U_use  = L(zero) - L(true), same
  right : U_pref split by A-2's liked colour (red-liked / green-liked episodes; descriptive, 6-pair means)
Style from analyze/paper/paper_style.py (Japanese font, Type 42), written to the working location, not paper/figs:
-> data/result/paper_extra/E2/fig_e2_upref.{pdf,png}
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

D = os.path.join(_PI, 'data', 'result', 'paper_extra', 'E2')
S = json.load(open(os.path.join(D, 'summary.json')))
CELL = 'pref|FP|1-50'
FAMS = [('none', '経験\nなし', PS.C_NONE), ('p50', '経験\n50%', PS.C_50), ('p80', '経験\n80%', PS.C_80),
        ('sr80', '符号\n無作為\n（80%）', PS.C_NS)]
DX = np.linspace(-0.27, 0.08, 6)


def save(fig, name):
    fig.savefig(os.path.join(D, f'{name}.png'), dpi=200)
    full = os.path.join(D, f'{name}_fullfont.pdf')
    fig.savefig(full)
    subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.5',
                    '-dSubsetFonts=true', '-dEmbedAllFonts=true', '-dAutoRotatePages=/None',
                    f'-sOutputFile={os.path.join(D, name + ".pdf")}', full], check=True)
    os.remove(full)
    print('wrote', os.path.join(D, name + '.{pdf,png}'))


def panel(ax, q, title):
    ax.axhline(0, color='0.6', lw=0.6, zorder=0)
    for i, (f, _, c) in enumerate(FAMS):
        A = S['alone'][CELL][f]
        for s in range(6):
            ax.plot(i + DX[s], A[q][s], 'o', ms=4, mec=c, mfc=c, alpha=0.85, zorder=3)
        t = A[f'{q}_t']
        ax.errorbar(i + 0.24, t['mean'], yerr=[[t['mean'] - t['ci95'][0]], [t['ci95'][1] - t['mean']]],
                    fmt='D', color='k', ms=4, capsize=2.5, lw=1.0, zorder=4)
    ax.set_xticks(range(len(FAMS)))
    ax.set_xticklabels([lab for _, lab, _ in FAMS])
    ax.set_xlim(-0.5, len(FAMS) - 0.5)
    ax.set_title(title)


def main():
    PS.setup()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw={'width_ratios': [4, 4, 4]})
    panel(axes[0], 'pref', '$U_{\\mathrm{pref}}$ = $L$(反) − $L$(正)')
    panel(axes[1], 'use', '$U_{\\mathrm{use}}$ = $L$(零) − $L$(正)')
    axes[0].set_ylabel('他者特徴の誤差の差\n（好み混合，フレーム1〜50，最終5点の平均）')
    ax = axes[2]
    ax.axhline(0, color='0.6', lw=0.6, zorder=0)
    for i, (f, _, c) in enumerate(FAMS):
        L = S['liked_split'][CELL][f]
        for j, (k, m) in enumerate((('red', 'o'), ('green', 's'))):
            x = i + (-0.15 if j == 0 else 0.15)
            ax.plot(x + np.linspace(-0.08, 0.08, 6), L[k], m, ms=3.2, mec=c, mfc='white' if k == 'green' else c, zorder=3)
            ax.plot([x - 0.11, x + 0.11], [np.mean(L[k])] * 2, '-', color='k', lw=1.2, zorder=4)
    ax.set_xticks(range(len(FAMS)))
    ax.set_xticklabels([lab for _, lab, _ in FAMS])
    ax.set_xlim(-0.5, len(FAMS) - 0.5)
    ax.set_title('$U_{\\mathrm{pref}}$：A-2 の好きな色別（記述）')
    h = [plt.Line2D([], [], ls='', marker='o', ms=4, mec='0.3', mfc='0.3', label='組1〜6（左・中）／赤好き（右）'),
         plt.Line2D([], [], ls='', marker='s', ms=3.5, mec='0.3', mfc='white', label='緑好き（右）'),
         plt.Line2D([], [], ls='-', marker='D', ms=4, color='k', label='6組の平均と 95% CI（$t$，自由度5）／右は6組の平均')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'fig_e2_upref')


if __name__ == '__main__':
    main()
