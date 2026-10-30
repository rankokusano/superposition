"""
Paper round 12, V-dir stage A (docs/paper_extra_log.md Sec.36; exploratory): figures from Vdir_A/summary.json
(vdir_stageA.py) only. Preference-mixed set, tau = 1.0, main regression.
  fig_vdir_coef : b_pref / b_dir / b_red (6-pair mean, t 95% CI = learning variation) per condition and window;
                  rows: oracle series (FP), c+- series (FP), oracle series (L1).
  fig_vdir_cells: Q_e per cell (A-2 liked x layout x direction of frames 1-50), per condition; oracle and c+- (FP).
Style from analyze/paper/paper_style.py (Japanese font, Type 42):
-> data/result/paper_extra/Vdir_A/fig_vdir_coef.{pdf,png}, fig_vdir_cells.{pdf,png}
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

D = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Vdir_A')
S = json.load(open(os.path.join(D, 'summary.json')))
CONDS = [('none', '経験なし', PS.C_NONE), ('p50', '経験50%', PS.C_50), ('p80', '経験80%', PS.C_80),
         ('sr80', '符号無作為（80%）', PS.C_NS), ('sr50', '符号無作為（50%）', '#8fd6bb'), ('sw', '好み切替（組1〜3）', '#9b59b6')]
WINS = [('1-25', '1〜25'), ('26-50', '26〜50'), ('51-100', '51〜100'), ('1-50', '1〜50'), ('1-100', '1〜100'),
        ('past', '過去の向き')]
TAU = '1.0'


def save(fig, name):
    fig.savefig(os.path.join(D, f'{name}.png'), dpi=200)
    full = os.path.join(D, f'{name}_fullfont.pdf')
    fig.savefig(full)
    subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.5',
                    '-dSubsetFonts=true', '-dEmbedAllFonts=true', '-dAutoRotatePages=/None',
                    f'-sOutputFile={os.path.join(D, name + ".pdf")}', full], check=True)
    os.remove(full)
    print('wrote', os.path.join(D, name + '.{pdf,png}'))


def coef_fig():
    rows = [('oracle', 'FP', 'oracle 系列・FP'), ('cpm', 'FP', '$c^{\\pm}$ 系列・FP'), ('oracle', 'L1', 'oracle 系列・L1')]
    terms = [('pref', '$b_{pref}$'), ('dir', '$b_{dir}$'), ('red', '$b_{red}$')]
    fig, axes = plt.subplots(3, 3, figsize=(7.4, 6.6), sharex=True)
    off = np.linspace(-0.3, 0.3, len(CONDS))
    for r, (series, err, rlab) in enumerate(rows):
        for c, (tm, tlab) in enumerate(terms):
            ax = axes[r, c]
            ax.axhline(0, color='0.6', lw=0.6, zorder=0)
            for i, (cd, lab, col) in enumerate(CONDS):
                for x, (w, _) in enumerate(WINS):
                    blk = S['coef'].get(f'{series}|{cd}|{err}|pref|{w}|{TAU}|main')
                    if blk is None:
                        continue
                    t = blk['t'][tm]
                    ax.plot(x + off[i], t['mean'], 'o', ms=3, color=col, label=lab if (x == 0) else None, zorder=3)
                    if t['ci95']:
                        ax.plot([x + off[i]] * 2, t['ci95'], '-', color=col, lw=0.9, zorder=2)
            ax.set_title(f'{rlab}：{tlab}', fontsize=7.5)
            ax.set_xticks(range(len(WINS)))
            ax.set_xticklabels([lab for _, lab in WINS], fontsize=6, rotation=40, ha='right')
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=6.5, ncol=6, loc='lower center')
    for ax in axes[:, 0]:
        ax.set_ylabel('係数（$Q_e$ の単位）')
    fig.suptitle('好み混合・τ = 1.0・主の回帰（6 組の平均と t の 95% CI＝学習の変動）。過去の向き＝向き 1〜25、$Q_e$ 26〜50', fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, 'fig_vdir_coef')


def cells_fig():
    cells = [(f'{lk}|L{L}|{dn}', f"{'赤' if lk == 'red' else '緑'}好き\n配置{L}\n向き{dn}")
             for lk in ('red', 'green') for L in (0, 1) for dn in ('A', 'B')]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.0))
    off = np.linspace(-0.3, 0.3, len(CONDS))
    for ax, (series, lab0) in zip(axes, (('oracle', 'oracle 系列'), ('cpm', '$c^{\\pm}$ 系列'))):
        ax.axhline(0, color='0.6', lw=0.6, zorder=0)
        C = {cd: S['cells'].get(f'{series}|{cd}|FP|pref|1-50|{TAU}') for cd, _, _ in CONDS}
        for i, (cd, lab, col) in enumerate(CONDS):
            if not C[cd]:
                continue
            for x, (ck, _) in enumerate(cells):
                if ck not in C[cd]:
                    continue
                t = C[cd][ck]['Q']
                ax.plot(x + off[i], t['mean'], 'o', ms=3, color=col, label=lab if x == 0 else None, zorder=3)
                if t['ci95']:
                    ax.plot([x + off[i]] * 2, t['ci95'], '-', color=col, lw=0.9, zorder=2)
        n = C['p50'] or C['none']
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{lab}\n(n={n[ck]['n']})" if ck in n else lab for ck, lab in cells], fontsize=6)
        ax.set_title(f'{lab0}・FP・フレーム 1〜50・τ = 1.0：セルごとの $Q_e$ ＝ L(B 快い) − L(A 快い)', fontsize=7.5)
        ax.set_ylabel('$Q_e$（6 組の平均、t の CI）')
        for x in (1.5, 3.5, 5.5):
            ax.axvline(x, color='0.85', lw=0.5, zorder=0)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=6.5, ncol=6, loc='lower center')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, 'fig_vdir_cells')


def main():
    PS.setup()
    coef_fig()
    cells_fig()


if __name__ == '__main__':
    main()
