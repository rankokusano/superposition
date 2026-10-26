"""
Paper extra A figures (docs/paper_instructions_extra.md Sec.6; not for paper/figs yet). From A_cross/summary.json:
  a_cross_delta : x = A-2 rule share (0.5 / 0.8), y = Delta (true - zero, other-side feature-prediction MSE, t = 1..99);
                  no experience / 50% / 80% as lines, thin = pairs (1-3 solid, 4-6 dotted), thick = mean of 6 pairs;
                  panels green-liking / mixed-preference data.
  a_cross_steps : Delta by A-2's step kind (follow / not follow / sleep) at 0.5 and 0.8, pairs 1-3 mean and range.
Saved with analyze/paper/paper_style (Type 42, IPAPGothic subset) into data/result/paper_extra/A_cross/figs/.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'paper'))
import paper_style as ps  # noqa

A = os.path.join(_PI, 'data', 'result', 'paper_extra', 'A_cross')
ps.FIGS = os.path.join(A, 'figs')
CONDS = (('random', ps.C_NONE, ps.M_NONE, ps.COND_LABEL['none']), ('mixed50', ps.C_50, ps.M_50, ps.COND_LABEL['50']),
         ('mixed80', ps.C_80, ps.M_80, ps.COND_LABEL['80']))
SETS = (('green', '(a) 緑好みデータ'), ('pref', '(b) 好み混合データ'))


def main():
    ps.setup()
    plt = ps.plt
    r = json.load(open(os.path.join(A, 'summary.json')))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, (k, title) in zip(axes, SETS):
        for c, col, mk, lab in CONDS:
            Y = np.array([[r['per_pair'][f'{k}_s{s}']['Delta'][f'{c}_{p}']['mean'] for p in ('0.5', '0.8')] for s in range(6)])
            for s in range(6):
                ax.plot([0, 1], Y[s], color=col, lw=0.6, alpha=0.55, ls='-' if s < 3 else ':')
            ax.plot([0, 1], Y.mean(0), color=col, lw=2.0, marker=mk, ms=5, label=lab)
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['0.5（元）', '0.8'])
        ax.set_xlim(-0.25, 1.25)
        ax.set_xlabel('A-2 が規則の一歩を選ぶ確率 $p$')
        ax.set_title(title)
    axes[0].set_ylabel('改善 Δ（真値 − 零）\n他者側の特徴予測の誤差')
    axes[0].legend(frameon=False, loc='lower left')
    fig.text(0.99, -0.1, '太線：組1〜6 の平均、細線：組ごと（実線 組1〜3、点線 組4〜6）。負ほど本当の快・不快が役立つ', ha='right', fontsize=7)
    ps.save(fig, 'a_cross_delta')

    kinds = (('follow', '従った一歩'), ('notfollow', '従わなかった一歩'), ('sleep', '眠り'))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True)
    for ax, (k, title) in zip(axes, SETS):
        for j, (c, col, mk, lab) in enumerate(CONDS):
            for pi, (kk, ls) in enumerate(((k, '--'), (k + '_a2p80', '-'))):
                V = np.array([[r['split'][f'{c}_s{s}_{kk}'][t]['delta'] for t, _ in kinds] for s in range(3)])
                x = np.arange(3) + (j - 1) * 0.22 + (pi - 0.5) * 0.09
                ax.errorbar(x, V.mean(0), yerr=[V.mean(0) - V.min(0), V.max(0) - V.mean(0)], color=col, fmt=mk,
                            mfc=col if pi else 'white', ms=4, lw=0.8, capsize=1.5,
                            label=f'{lab}・{"0.8" if pi else "0.5"}')
        ax.axhline(0, color='0.6', lw=0.5)
        ax.set_xticks(range(3)); ax.set_xticklabels([n for _, n in kinds])
        ax.set_xlabel('A-2 の一歩の種類')
        ax.set_title(title)
    axes[0].set_ylabel('改善 Δ（真値 − 零）')
    axes[1].legend(frameon=False, fontsize=6.5, ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.text(0.99, -0.1, '組1〜3 の平均と範囲。白抜き：A-2 0.5（元のデータ）、塗り：A-2 0.8', ha='right', fontsize=7)
    ps.save(fig, 'a_cross_steps')


if __name__ == '__main__':
    main()
