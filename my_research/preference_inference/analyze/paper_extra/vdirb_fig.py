"""
Paper round 13 rev2, V-dir stage B (docs/paper_extra_log.md Sec.37.5.7): figures. Descriptive only.
  fig_vdirb_coef : b_pref / b_dir (6-pair mean, t 95% CI; two-stage CI as a thin bar) per condition, main regression
                   and sensitivity forms, c+- FP frames 1-50 tau 1.0 (from Vdir_B/analysis/summary.json).
  fig_vdirb_cells: Q_e per cell (layout x liked x direction of frames 1-50, tau 1.0), per condition, c+- FP
                   (6-pair mean of the 5-point means of the cell mean; t CI), from the evaluation arrays.
Style from analyze/paper/paper_style.py (Type 42). -> data/result/paper_extra/Vdir_B/analysis/fig_vdirb_*.{pdf,png}
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
import vdirb_analysis as A  # noqa
plt = PS.plt

D = A.OUT
S = json.load(open(os.path.join(D, 'summary.json')))
CONDS = [('none', '経験なし', PS.C_NONE), ('p50', '経験50%', PS.C_50), ('p80', '経験80%', PS.C_80),
         ('sr80', '符号無作為（80%）', PS.C_NS), ('sr50', '符号無作為（50%）', '#8fd6bb'), ('sw', '好み切替（組1〜3）', '#9b59b6')]
FORMS = [('main', '主'), ('s_i', '(i) x 座標'), ('s_ii', '(ii) 角までの距離'), ('s_iii', '(iii) 初期位置なし'),
         ('s_iv', '(iv) 3 区分 × 向き'), ('s_v', '(v) A 側の割合'), ('nodir', '向きを調整しない')]


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
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=False)
    off = np.linspace(-0.32, 0.32, len(CONDS))
    for ax, (tm, tl) in zip(axes, (('pref', '$b_{pref}$'), ('dir', '$b_{dir}$（(v) は A 側の割合の係数）'))):
        ax.axhline(0, color='0.6', lw=0.6, zorder=0)
        for i, (cd, lab, col) in enumerate(CONDS):
            for x, (fm, _) in enumerate(FORMS):
                b = S['coef'].get(f'cpm|FP|1-50|1.0|{fm}|{cd}')
                if b is None or tm not in b['t']:
                    continue
                t = b['t'][tm]
                ax.plot(x + off[i], t['mean'], 'o', ms=3, color=col, label=lab if x == 0 else None, zorder=3)
                if t['ci95']:
                    ax.plot([x + off[i]] * 2, t['ci95'], '-', color=col, lw=1.0, zorder=2)
                b2 = b.get('boot2', {}).get(tm)
                if b2:
                    ax.plot([x + off[i] + 0.03] * 2, b2, '-', color=col, lw=0.4, alpha=0.7, zorder=2)
        ax.set_xticks(range(len(FORMS)))
        ax.set_xticklabels([l for _, l in FORMS], fontsize=6, rotation=40, ha='right')
        ax.set_title(tl, fontsize=8)
    axes[0].set_ylabel('係数（$Q_e$ の単位）')
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=6.5, ncol=6, loc='lower center')
    fig.suptitle('段階B：$c^{\\pm}$・FP・フレーム 1〜50・τ = 1.0（点＝組の平均、太線＝6 組の t の 95% CI、細線＝2段のブートストラップ）', fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, 'fig_vdirb_coef')


def cells_fig():
    m = A.meta()
    p = m['p']['1-50']
    keep = (np.abs(p) >= 1.0) & (p != 0)
    sd = np.sign(p)
    cells = [(lk, L, dn) for lk in (1, 0) for L in (0, 1) for dn in (1, -1)]
    labs = [f"{'赤' if lk else '緑'}好き\n配置{L}\n向き{'A' if dn > 0 else 'B'}" for lk, L, dn in cells]
    ns = [int(((m['liked'] == lk) & (m['layout'] == L) & keep & (sd == dn)).sum()) for lk, L, dn in cells]
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 3.4))
    ax.axhline(0, color='0.6', lw=0.6, zorder=0)
    off = np.linspace(-0.32, 0.32, len(CONDS))
    for i, (cd, lab, col) in enumerate(CONDS):
        q = A.load_q('cpm', 'FP', cd, m)['1-50']                      # (np, 5, n)
        for x, (lk, L, dn) in enumerate(cells):
            sel = (m['liked'] == lk) & (m['layout'] == L) & keep & (sd == dn)
            v = q[:, :, sel].mean(2).mean(1)
            t = A.tsum(v)
            ax.plot(x + off[i], t['mean'], 'o', ms=3, color=col, label=lab if x == 0 else None, zorder=3)
            if t['ci95']:
                ax.plot([x + off[i]] * 2, t['ci95'], '-', color=col, lw=0.9, zorder=2)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(labs, ns)], fontsize=6)
    for x in (1.5, 3.5, 5.5):
        ax.axvline(x, color='0.85', lw=0.5, zorder=0)
    ax.set_ylabel('$Q_e$ ＝ L($c^-$) − L($c^+$)')
    ax.set_title('段階B：セルごとの $Q_e$（$c^{\\pm}$・FP・フレーム 1〜50・τ = 1.0。組の平均と t の 95% CI）', fontsize=7.5)
    h, lb = ax.get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=6.5, ncol=6, loc='lower center')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'fig_vdirb_cells')


def main():
    PS.setup()
    coef_fig()
    cells_fig()


if __name__ == '__main__':
    main()
