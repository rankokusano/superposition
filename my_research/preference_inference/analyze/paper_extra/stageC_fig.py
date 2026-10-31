"""
Paper round 14 (stage C) figures from StageC/analysis/summary.json (stageC_analysis.py) only (docs/paper_extra_log.md
Sec.38.5.8; instructions Sec.4.5). Descriptive. Type 42 via ghostscript as the other paper figures.
  fig_stageC_a : b^q_pref (standardised q_e) stage 1 and stage 2 per pair (lines join the pairs), 50% / none /
                 pre-training; mean and t CI.
  fig_stageC_b : read-outs (nested C): basic BA and adjusted Delta LL for O, S, Q, H, H0; 50% / none / pre-training.
  fig_stageC_c : generalisation (agree -> disagreeing; layout 0 -> 1, 1 -> 0) per representation.
  fig_stageC_d : gradient projection g_e (total loss, point G, steps 1-50): b_pref stage 1 / stage 2 / (v) form.
-> data/result/paper_extra/StageC/analysis/fig_stageC_{a,b,c,d}.{pdf,png}
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
D = os.path.join(_PI, 'data', 'result', 'paper_extra', 'StageC', 'analysis')
S = json.load(open(os.path.join(D, 'summary.json')))
COND = [('p50', '経験50%', PS.C_50), ('none', '経験なし', PS.C_NONE), ('pre', '学習前（経験50% の段階I）', '#9b59b6')]
REPS = [('O', '(O) 観察'), ('S', '(S) G の状態'), ('Q', '(Q) $Q2$'), ('H', '(H) $h^2$'), ('H0', '(H0) $h^2$, $Q2$=0')]


def save(fig, name):
    fig.savefig(os.path.join(D, f'{name}.png'), dpi=200)
    full = os.path.join(D, f'{name}_fullfont.pdf')
    fig.savefig(full)
    subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.5',
                    '-dSubsetFonts=true', '-dEmbedAllFonts=true', '-dAutoRotatePages=/None',
                    f'-sOutputFile={os.path.join(D, name + ".pdf")}', full], check=True)
    os.remove(full)


def fig_a():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, (form, lab) in zip(axes, (('s1', '第1段（調整しない、1200）'), ('s2', '第2段（向きを調整、1088）'))):
        ax.axhline(0, color='0.6', lw=0.6)
        for i, (k, nm, col) in enumerate(COND):
            e = S['L1'][f'{k}|{form}|std|pref']['95']['t']
            x = i + np.zeros(len(e['per_pair']))
            ax.plot(x, e['per_pair'], 'o', color=col, ms=3, alpha=0.7)
            ax.errorbar(i + 0.15, e['mean'], yerr=[[e['mean'] - e['ci'][0]], [e['ci'][1] - e['mean']]], fmt='s', color=col, ms=4, capsize=2)
        pp = np.array([S['L1'][f'{k}|{form}|std|pref']['95']['t']['per_pair'] for k, _, _ in COND])
        for j in range(pp.shape[1]):
            ax.plot(range(len(COND)), pp[:, j], '-', color='0.7', lw=0.6, zorder=0)
        ax.set_xticks(range(len(COND)))
        ax.set_xticklabels([nm for _, nm, _ in COND], fontsize=7)
        ax.set_title(lab, fontsize=8)
    axes[0].set_ylabel('$b^{q}_{pref}$（標準化した $q_e$）')
    fig.suptitle('段階C (L1)：組ごとの値（線で結ぶ）と 4 組の平均・t の 95% CI', fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_stageC_a')


def bar(ax, metric, ylabel, ref=None):
    off = np.linspace(-0.25, 0.25, len(COND))
    for i, (k, nm, col) in enumerate(COND):
        for x, (r, _) in enumerate(REPS):
            e = S['readout'][f'nest|{r}|{k}|{metric}']
            ax.plot(x + off[i], e['t']['mean'], 'o', color=col, ms=3.5, label=nm if x == 0 else None)
            ax.plot([x + off[i]] * 2, e['t']['ci'], '-', color=col, lw=1.0)
            if e['boot']:
                ax.plot([x + off[i] + 0.04] * 2, e['boot'], '-', color=col, lw=0.4, alpha=0.7)
    if ref is not None:
        ax.axhline(ref, color='0.6', lw=0.6)
    ax.set_xticks(range(len(REPS)))
    ax.set_xticklabels([l for _, l in REPS], fontsize=6.5, rotation=20)
    ax.set_ylabel(ylabel, fontsize=7.5)


def fig_b():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    bar(axes[0], 'BA', '好みの BA（基本の読み出し、1200）', 0.5)
    bar(axes[1], 'dLL', '$\\Delta$LL（調整した読み出し、1088）', 0.0)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=7, ncol=3, loc='lower center')
    fig.suptitle('段階C (L2)(L3)：入れ子の交差検証の C。太線＝4 組の t の 95% CI、細線＝2段。(L3) の $\\Delta$LL は記述（雑音でも負にずれる）', fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, 'fig_stageC_b')


def fig_c():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), sharey=True)
    for ax, (m, lab) in zip(axes, (('gen_dis', '一致 → 食い違う 280'), ('gen_lay01', '配置0 → 配置1'), ('gen_lay10', '配置1 → 配置0'))):
        bar(ax, m, '好みの正解率' if m == 'gen_dis' else '', 0.5)
        ax.set_title(lab, fontsize=8)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=7, ncol=3, loc='lower center')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, 'fig_stageC_c')


def fig_d():
    keys = [('p50', '経験50%'), ('none', '経験なし'), ('pre', '学習前'), ('p80free', '自由出力・経験80%'),
            ('green50', '緑好み・経験50%'), ('onebit50', '1 ビット・経験50%'), ('onebit80', '1 ビット・経験80%')]
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.0))
    ax.axhline(0, color='0.6', lw=0.6)
    off = {'s1': -0.2, 's2': 0.0, 'v': 0.2}
    cols = {'s1': '#333333', 's2': PS.C_50, 'v': PS.C_80}
    for x, (k, nm) in enumerate(keys):
        for form in ('s1', 's2', 'v'):
            e = S['descriptive'][k][f'grad_b_pref_{form}']
            ax.plot(x + off[form], e['mean'], 'o', color=cols[form], ms=3.5,
                    label={'s1': '第1段の形', 's2': '第2段の形', 'v': '(v) の形'}[form] if x == 0 else None)
            ax.plot([x + off[form]] * 2, e['ci'], '-', color=cols[form], lw=1.0)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([nm for _, nm in keys], fontsize=6.5, rotation=20)
    ax.set_ylabel('$g_e$ の $b_{pref}$（標準化しない）', fontsize=7.5)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title('段階C 勾配（探索）：$-\\sum_{t=1}^{50}\\partial L/\\partial Q2_t\\cdot c^+/|c^+|$（全体の損失、G の出力の点）。t の 95% CI', fontsize=7.5)
    fig.tight_layout()
    save(fig, 'fig_stageC_d')


def main():
    PS.setup()
    fig_a(); fig_b(); fig_c(); fig_d()
    print('ok')


if __name__ == '__main__':
    main()
