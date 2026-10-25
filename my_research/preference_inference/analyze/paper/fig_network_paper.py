"""
Paper figure 1 'network' (documents only): the network and the two training stages, redrawn from
docs/figs/network_v11_v12.svg (v12 part only) with the paper's names. Facts drawn here were checked against the code
(paper/check/methods_check.md): encoders frozen at the reference weights; the visual predictor = integration layer
(cat[h1, h2] -> 64) + decoder, and the L1 period trains only the decoder; process-1 input [motion 2, pleasure 8,
hidden flag 1]; process-2 input zero in stage I and [0, Q2, 1] in stage II; G reads the unmasked encoder-2 output;
no-sharing condition splits only the LSTM.
Writes paper/figs/network.{pdf,png}.
"""

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_style import plt, save, setup

TRAIN = 'white'
FROZEN = '#e6e6e6'
HATCH = '////'
EC = '0.25'


def box(ax, x, y, w, h, text, frozen=False, fs=7, ec=EC, lw=0.8, fc=None, style='round,pad=0.02,rounding_size=0.12',
        weight='normal', z=2):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style, fc=fc or (FROZEN if frozen else TRAIN), ec=ec,
                       lw=lw, hatch=HATCH if frozen else None, zorder=z)
    ax.add_patch(p)
    if text:
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, zorder=z + 1, weight=weight,
                bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.85) if frozen else None)
    return p


def arrow(ax, a, b, color='0.2', lw=0.8, ls='-', z=6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=7, color=color, lw=lw, ls=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def panel(ax, stage):
    s2 = stage == 2
    ax.set_xlim(-0.2, 12.4)
    ax.set_ylim(0, 10.3)
    ax.axis('off')
    box(ax, 6, 0.7, 3.8, 0.9, 'A-1 の視界 $v_t$（16×64，全方位）', fc='#fafafa')
    for x, n in ((4.5, 1), (7.5, 2)):
        box(ax, x, 2.4, 2.7, 0.9, f'視覚エンコーダ{n}', frozen=True)
        arrow(ax, (6 + (x - 6) * 0.5, 1.15), (x, 1.95))
        arrow(ax, (x, 2.85), (x, 3.93))
        ax.text(x + 0.08, 3.38, 'マスク', fontsize=5.8, ha='left', va='center', color='0.3')
    # shared module
    box(ax, 6, 4.75, 6.6, 2.1, None, frozen=s2, z=1)
    ax.text(6, 5.55, '共有モジュール（LSTM 128，process-1・2 で重みを共有）', ha='center', va='center', fontsize=6.6,
            zorder=4, bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85) if s2 else None)
    box(ax, 4.5, 4.4, 2.7, 0.95, 'process-1\n内部状態 $h^1_t$', frozen=s2, fs=6.8, z=3)
    box(ax, 7.5, 4.4, 2.7, 0.95, 'process-2\n内部状態 $h^2_t$', frozen=s2, fs=6.8, z=3)
    # process-1 input
    box(ax, 1.05, 4.4, 2.0, 1.55, '$x^1_t$\n運動 2\n快・不快 8\n隠れフラグ 1', fs=6.4, fc='#fff6ee', ec='#c45a1c')
    arrow(ax, (2.05, 4.4), (3.15, 4.4), color='#c45a1c')
    # process-2 input
    if s2:
        box(ax, 10.95, 4.4, 2.3, 1.0, '$x^2_t = [0, Q2_t, 1]$', fs=6.4, fc='#fff6ee', ec='#c45a1c')
        box(ax, 10.95, 2.4, 2.6, 1.1, '他者用モジュール $G$\nLSTMCell 128 → 8，tanh', fs=6.3, ec='#1baf7a', lw=1.4)
        arrow(ax, (8.85, 2.15), (9.75, 2.15), color='#1baf7a', lw=1.0)
        ax.text(9.3, 1.75, 'マスク前', fontsize=5.6, ha='center', color='#1baf7a')
        arrow(ax, (10.95, 2.95), (10.95, 3.9), color='#1baf7a', lw=1.0)
        ax.text(11.05, 3.42, '$Q2_t$', fontsize=6.5, ha='left', va='center', color='#1baf7a')
    else:
        box(ax, 10.95, 4.4, 2.1, 1.0, '$x^2_t$\n零（11 次元）', fs=6.4, fc='#fff6ee', ec='#c45a1c')
    arrow(ax, (9.9, 4.4), (8.85, 4.4), color='#c45a1c')
    # visual predictor
    box(ax, 6, 7.15, 5.0, 1.55, None, frozen=s2, z=1)
    ax.text(6, 7.67, '視覚予測器', ha='center', va='center', fontsize=6.8, zorder=4,
            bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85) if s2 else None)
    box(ax, 4.85, 7.0, 2.1, 0.75, '統合層', frozen=s2, fs=6.6, z=3)
    box(ax, 7.15, 7.0, 2.1, 0.85, '復号部' if s2 else '復号部\n（L1 期間はここだけ学習）', frozen=s2,
        fs=6.6 if s2 else 6.0, z=3)
    arrow(ax, (5.9, 7.0), (6.1, 7.0))
    arrow(ax, (4.5, 4.88), (4.6, 6.62))
    arrow(ax, (7.5, 4.88), (5.3, 6.62))
    box(ax, 7.15, 9.35, 2.9, 0.75, '予測 $\\hat{v}_{t+1}$（A-1 の次の視界）', fs=6.3, fc='#fafafa')
    arrow(ax, (7.15, 7.38), (7.15, 8.97))
    # feature predictor
    box(ax, 10.95, 7.0, 2.4, 1.15, '特徴予測器\n（次の視覚エンコーダ\n出力を予測）', frozen=s2, fs=6.0)
    arrow(ax, (9.3, 5.4), (10.4, 6.42))
    if not s2:
        # inset: no-sharing condition
        box(ax, 1.55, 8.55, 3.0, 2.15, None, fc='#f7fbff', ec='#2a78d6', lw=0.9, z=1)
        ax.text(1.55, 9.35, '差し込み：非共有条件', fontsize=6.3, ha='center', color='#2a78d6')
        box(ax, 0.8, 8.4, 1.35, 0.85, 'LSTM\n（process-1）', fs=5.6, ec='#2a78d6', z=3)
        box(ax, 2.3, 8.4, 1.35, 0.85, 'LSTM\n（process-2）', fs=5.6, ec='#2a78d6', z=3)
        ax.text(1.55, 7.65, '重みを共有しない', fontsize=5.6, ha='center', color='#2a78d6')
    ax.set_title('(a) 段階I：自己の予測学習' if not s2 else '(b) 段階II：他者用モジュールの学習（$G$ 以外は凍結）',
                 fontsize=8.5, loc='left')


def main():
    setup()
    fig, axs = plt.subplots(2, 1, figsize=(6.3, 6.4))
    panel(axs[0], 1)
    panel(axs[1], 2)
    hs = [FancyBboxPatch((0, 0), 1, 1, fc=TRAIN, ec=EC), FancyBboxPatch((0, 0), 1, 1, fc=FROZEN, ec=EC, hatch=HATCH)]
    fig.legend(hs, ['その段階で学習する', '凍結（視覚エンコーダは基準エンコーダの重み）'], loc='lower center', ncol=2,
               frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=0.8)
    save(fig, 'network')


if __name__ == '__main__':
    main()
