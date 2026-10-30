"""
Paper round 11 (docs/paper_extra_log.md Sec.35.12 C): per-frame U_pref on the preference-mixed set, FP, from
E2/episode/summary.json (e2_episode.py) only.
  left  : per-frame U_f = FP_opp - FP_true (6-pair mean, t 95% CI band), 4 conditions
  middle: 50% and 80% split by A-2's liked colour (6-pair means)
  right : cumulative-window U(1..T) for T = 1..100 (the E2 quantity for the window 1-T), 4 conditions; T = 50 marked
Style from analyze/paper/paper_style.py (Japanese font, Type 42):
-> data/result/paper_extra/E2/episode/fig_e2_frames.{pdf,png}
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

D = os.path.join(_PI, 'data', 'result', 'paper_extra', 'E2', 'episode')
C = json.load(open(os.path.join(D, 'summary.json')))['C']
FAMS = [('none', '経験なし', PS.C_NONE), ('p50', '経験50%', PS.C_50), ('p80', '経験80%', PS.C_80),
        ('sr80', '符号無作為（80%）', PS.C_NS)]
F = np.arange(1, 101)


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
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ax = axes[0]
    for f, lab, c in FAMS:
        r = C[f]['U']
        ax.fill_between(F, r['ci_lo'], r['ci_hi'], color=c, alpha=0.18, lw=0)
        ax.plot(F, r['mean'], color=c, lw=1.1, label=lab)
    ax.set_title('フレームごとの $U_{pref}$')
    ax.legend(frameon=False, fontsize=6.5, loc='upper right')
    ax = axes[1]
    for f, lab, c in FAMS[1:3]:
        ax.plot(F, C[f]['red']['mean'], color=c, lw=1.1, label=f'{lab}・A-2 赤好き')
        ax.plot(F, C[f]['green']['mean'], color=c, lw=1.1, ls='--', label=f'{lab}・A-2 緑好き')
    ax.set_title('A-2 の好き別（6 組の平均）')
    ax.legend(frameon=False, fontsize=6, loc='lower left')
    ax = axes[2]
    for f, lab, c in FAMS:
        r = C[f]['cum']
        ax.fill_between(F, r['ci_lo'], r['ci_hi'], color=c, alpha=0.18, lw=0)
        ax.plot(F, r['mean'], color=c, lw=1.1, label=lab)
    ax.axvline(50, color='0.5', lw=0.6, ls=':')
    ax.set_title('区間 1〜T の $U_{pref}$')
    for i, ax in enumerate(axes):
        ax.axhline(0, color='0.6', lw=0.6, zorder=0)
        ax.set_xlim(1, 100)
        ax.set_xlabel('フレーム' if i < 2 else 'T')
    axes[0].set_ylabel('FP(反) − FP(正)')
    fig.tight_layout()
    save(fig, 'fig_e2_frames')


if __name__ == '__main__':
    main()
