"""
Paper round 11, E1 (docs/paper_extra_log.md Sec.35.11): per-pair late-5 means of none / 50% / 50% sign-randomised
(signrand50) from E1_signrand50/summary.json (e1_aggregate.py) and E2/summary.json (e2_aggregate.py) only.
  panels: d_P (main), sign-following (1) green, (2) pref, E2 U_pref (pref, FP, frames 1-50). Pairs 1-6 as dots, the
          50% and signrand50 values of the same pair joined by a thin line; the 6-pair difference signrand50 - 50%
          (t 95% CI, df 5) written under each panel title.
Style from analyze/paper/paper_style.py (Japanese font, Type 42), written to the working location, not paper/figs:
-> data/result/paper_extra/E1_signrand50/fig_e1.{pdf,png}
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

RES = os.path.join(_PI, 'data', 'result', 'paper_extra')
D = os.path.join(RES, 'E1_signrand50')
S = json.load(open(os.path.join(D, 'summary.json')))
E2S = json.load(open(os.path.join(RES, 'E2', 'summary.json')))
CELL = 'pref|FP|1-50'
MODELS = [('none', '経験\nなし', PS.C_NONE), ('50', '経験\n50%', PS.C_50), ('sr50', '符号\n無作為\n（50%）', PS.C_NS)]


def save(fig, name):
    fig.savefig(os.path.join(D, f'{name}.png'), dpi=200)
    full = os.path.join(D, f'{name}_fullfont.pdf')
    fig.savefig(full)
    subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.5',
                    '-dSubsetFonts=true', '-dEmbedAllFonts=true', '-dAutoRotatePages=/None',
                    f'-sOutputFile={os.path.join(D, name + ".pdf")}', full], check=True)
    os.remove(full)
    print('wrote', os.path.join(D, name + '.{pdf,png}'))


def vals(q):
    if q == 'e2':
        e2 = {'none': E2S['alone'][CELL]['none']['pref'], '50': E2S['alone'][CELL]['p50']['pref'],
              'sr50': S['e2'][CELL]['sr50_alone']['pref']}
        return e2
    return {m: [float(np.mean(S['values'][f's{s}'][m][q])) for s in range(6)] for m, _, _ in MODELS}


def diff_t(q):
    t = S['e2'][CELL]['p50_minus_sr50']['t'] if q == 'e2' else S['across'][q]['t_means']
    sgn = -1 if q == 'e2' else 1   # written as signrand50 - 50% in every panel
    m, lo, hi = sgn * t['mean'], sgn * t['ci95'][1 if q == 'e2' else 0], sgn * t['ci95'][0 if q == 'e2' else 1]
    n = 4 if q == 'e2' else 3
    return f'差（符号無作為 − 50%）\n{m:+.{n}f} [{lo:+.{n}f}, {hi:+.{n}f}]'


def panel(ax, q, title):
    V = vals(q)
    ax.axhline(0, color='0.6', lw=0.6, zorder=0)
    for s in range(6):
        ax.plot([1, 2], [V['50'][s], V['sr50'][s]], '-', color='0.7', lw=0.6, zorder=1)
    for i, (m, _, c) in enumerate(MODELS):
        ax.plot(np.full(6, i), V[m], 'o', ms=4, mec=c, mfc=c, alpha=0.85, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [np.mean(V[m])] * 2, '-', color='k', lw=1.2, zorder=4)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([lab for _, lab, _ in MODELS])
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_title(f'{title}\n{diff_t(q)}', fontsize=7.5)


def main():
    PS.setup()
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.9))
    for ax, (q, t) in zip(axes, (('dP', '主：$d_P$'), ('signfollow_green', '符号に従う分 ①（緑好み）'),
                                  ('signfollow_pref', '符号に従う分 ②（好み混合）'), ('e2', 'E2 の $U_{pref}$（FP, 1〜50）'))):
        panel(ax, q, t)
    fig.tight_layout()
    save(fig, 'fig_e1')


if __name__ == '__main__':
    main()
