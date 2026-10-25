"""
v15 stage-1 figures (pre-registered in docs/v15_experiment_log.md Sec.10.2 / 12.4), from the training logs,
data/result/v15/stage1/*.json (eval_s1_v15.py) and the Fig.5 outputs (fig5_v11.py). CPU.

    python analyze/v15/figs_s1_v15.py    # -> data/result/v15/stage1/figs/
"""

import json
import os
import shutil
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import torch  # noqa

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
sys.path.insert(0, os.path.join(_PI_DIR, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI_DIR, 'analyze', 'v11'))
from fig5_extra_v11 import make_ae  # noqa
from fig5_colour_check_v12 import rec, sep  # noqa
from analyze.v15.eval_s1_v15 import DATA, TS, sep_bc  # noqa

RES = os.path.join(_PI_DIR, 'data', 'result')
S1 = os.path.join(RES, 'v15', 'stage1')
FIG = os.path.join(S1, 'figs')
RUNS = {'plain': '(i) usual loss', 'colour': '(ii) + colour (chroma) loss'}
COL = {'v4': '#9a9a94', 'E1': '#eda100', 'plain': '#2a78d6', 'colour': '#eb6834'}
INK = '#3d3d3a'
# reference Fig.5 margins from docs/v12_experiment_log.md (v4: Sec.8, original layout only; E1 mse ep200: Sec.53)
REF_MARGIN = {'v4': (0.091, np.nan), 'E1': (0.075, 0.075)}


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8)


def read_log(path):
    if not os.path.exists(path):
        return np.array([]), np.array([])
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    return a[:, 0], a[:, 1]


def fig_curves():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for j, (st, lab) in enumerate((('mse', 'mse stage (whole network)'), ('l1', 'l1 stage (Decoder only)'))):
        for run in RUNS:
            base = os.path.join(RES, f'v15_s1_{run}_{st}', '0', 'log')
            for split, ls in (('train', '-'), ('test', '--')):
                e, v = read_log(os.path.join(base, split, 'self_vision.log'))
                ax[j].plot(e, v, ls, color=COL[run], lw=1.6, label=f'{RUNS[run]} {split}: view loss')
            if run == 'colour':
                e, v = read_log(os.path.join(base, 'train', 'self_vision_colour.log'))
                ax[j].plot(e, v, ':', color=COL[run], lw=1.6, label=f'{RUNS[run]} train: colour term (x10)')
        ax[j].set_yscale('log'); ax[j].set_xlabel('epoch'); ax[j].set_ylabel('loss (sum over steps, log scale)')
        ax[j].set_title(lab, fontsize=9); style(ax[j])
    ax[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Stage 1 learning curves (seed 0)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 's1_learning_curves.png'), dpi=120); plt.close(fig)


def load_eval(tag):
    return json.load(open(os.path.join(S1, f'{tag}.json')))


def fig5_margin(run):
    r = json.load(open(os.path.join(S1, f'fig5_{run}_mse200', f'fig5_v15_s1_{run}_fig5_mse200_s0_ep100.json')))
    return r['original']['margin_other_self_minus_other_other'], r['swap_rg']['margin_other_self_minus_other_other']


def rows():
    return [('v4', load_eval('ref_v4'), REF_MARGIN['v4']), ('E1', load_eval('ref_e1_mse200'), REF_MARGIN['E1'])] + \
           [(run, load_eval(f'v15_{run}_mse200'), fig5_margin(run)) for run in RUNS]


def fig_alignment(R):
    labels = {'v4': 'v4 (frozen in v12)', 'E1': 'E1 (v12, paper model, v11 data)', **RUNS}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(R))
    a1 = [r['M1_alignment_v15_test'] for _, r, _ in R]
    a2 = [r['M1_alignment_v11_common_test']['cos_ov(A1)_sv(A2)'] for _, r, _ in R]
    ax[0].bar(x - 0.2, a1, 0.38, color=[COL[k] for k, _, _ in R], label='v15 test')
    ax[0].bar(x + 0.2, a2, 0.38, color=[COL[k] for k, _, _ in R], alpha=0.45, label='v11 common test')
    ax[0].axhline(0.70, color=INK, ls='--', lw=1); ax[0].text(len(R) - 0.5, 0.71, 'pass >= 0.70', fontsize=7, ha='right')
    for i, (v, w) in enumerate(zip(a1, a2)):
        ax[0].text(i - 0.2, v + 0.01, f'{v:.2f}', ha='center', fontsize=7); ax[0].text(i + 0.2, w + 0.01, f'{w:.2f}', ha='center', fontsize=7)
    ax[0].set_ylabel('cos_ov(A1)_sv(A2)'); ax[0].set_title('self/other alignment (M1); dark = v15 test, light = v11 common test', fontsize=9)
    m = np.array([mg for _, _, mg in R])
    ax[1].bar(x - 0.2, m[:, 0], 0.38, color=[COL[k] for k, _, _ in R])
    ax[1].bar(x + 0.2, m[:, 1], 0.38, color=[COL[k] for k, _, _ in R], alpha=0.45)
    for i in range(len(R)):
        for dx, v in ((-0.2, m[i, 0]), (0.2, m[i, 1])):
            if np.isfinite(v):
                ax[1].text(i + dx, v + 0.002, f'{v:+.3f}', ha='center', fontsize=7)
    ax[1].set_ylabel('Fig.5 margin (other_self - other_other)')
    ax[1].set_title('Fig.5 margin; dark = original, light = swapped grid (v4: original only)', fontsize=9)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([labels[k] for k, _, _ in R], fontsize=7, rotation=10); style(a)
    fig.suptitle('Stage 1: alignment and Fig.5 margin at mse ep200 (v4, E1 = references)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 's1_alignment_margin.png'), dpi=120); plt.close(fig)


def fig_corners(R):
    corners = ('TL', 'BL', 'BR', 'TR')
    names = {'TL': 'top-left {R,G,gray}', 'BL': 'bottom-left {R,G,gray}', 'BR': 'bottom-right {B,gray}', 'TR': 'top-right {C,gray}'}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    w = 0.2
    for j, enc in enumerate(('enc2', 'enc1')):
        for i, (k, r, _) in enumerate(R):
            v = [r['M3'][c][enc] for c in corners]
            ax[j].bar(np.arange(4) + (i - 1.5) * w, v, w * 0.9, color=COL[k], label=k if k in ('v4', 'E1') else RUNS[k])
        maj = [R[0][1]['M3'][c]['majority'] for c in corners]
        for c in range(4):
            ax[j].hlines(maj[c], c - 0.45, c + 0.45, color=INK, lw=1.2, ls=':')
        if enc == 'enc2':
            ax[j].axhline(0.90, color=INK, ls='--', lw=1)
        ax[j].set_xticks(range(4)); ax[j].set_xticklabels([names[c] for c in corners], fontsize=8)
        ax[j].set_ylim(0.4, 1.02); ax[j].set_title(('Encoder-2' if enc == 'enc2' else 'Encoder-1') +
                                                  ' (A-1 view); dotted = majority class, dashed = pass 0.90 (Encoder-2)', fontsize=8)
        style(ax[j])
    ax[0].set_ylabel('linear readout accuracy (5-fold, by episode)'); ax[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Stage 1 (M3): corner state read from the encoders, mse ep200', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 's1_corner_readout.png'), dpi=120); plt.close(fig)


@torch.no_grad()
def fig_images(run):
    g = h5py.File(DATA, 'r')['test']
    lay, gray0 = g['layout'][()], g['gray0'][()]
    m, _ = make_ae(os.path.join(RES, f'v15_s1_{run}_mse', '0', 'model', '00200.pth'),
                   os.path.join(RES, f'v15_s1_{run}_fig5_mse200', '0', 'model', '00100.pth'))
    gray_names = {-1: 'none', 0: 'Red', 1: 'Green', 2: 'Blue', 3: 'Cyan'}
    picks = []
    for L in (0, 1):
        for gc in (-1, 0, 1, 2, 3):
            eps = np.where((lay == L) & (gray0 == gc))[0]
            if len(eps):
                picks.append((L, gc, eps[0], 10))   # frame t = 10 of the first such test episode
    sv = np.stack([g['self_vision'][e, t] for _, _, e, t in picks])
    ov = np.stack([g['other_vision'][e, t] for _, _, e, t in picks])
    rs, ro = rec(m, sv), rec(m, sv, other=True)
    # sep over all test frames of this run (as in M4)
    r = load_eval(f'v15_{run}_mse200')['M4']
    fig, ax = plt.subplots(len(picks), 4, figsize=(14, 1.25 * len(picks) + 0.8))
    for i, (L, gc, e, t) in enumerate(picks):
        for k, (img, lab) in enumerate(((sv[i], 'A-1 view (input)'), (rs[i], 'self reconstruction D(E1)'),
                                        (ov[i], 'A-2 true view'), (ro[i], 'Fig.5 view D(E2)'))):
            ax[i, k].imshow(img if img.dtype == np.uint8 else (img * 255).astype(np.uint8), interpolation='nearest')
            ax[i, k].set_xticks([]); ax[i, k].set_yticks([])
            if i == 0:
                ax[i, k].set_title(lab, fontsize=8)
        ax[i, 0].set_ylabel(f"{'no swap' if L == 0 else 'swap'}\ngray: {gray_names[gc]}", fontsize=7)
    fig.suptitle(f"{RUNS[run]}, mse ep200. Test episodes, t=10. Over all test frames: Fig.5 view sep R/G {r['other_sepRG']:+.2f}, "
                 f"B/C {r['other_sepBC']:+.2f}; self rec sep R/G {r['self_sepRG']:+.2f}, B/C {r['self_sepBC']:+.2f} "
                 f"(true view {r['true_sepRG']:.2f} / {r['true_sepBC']:.2f})", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f's1_fig5_images_{run}.png'), dpi=110); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    torch.set_num_threads(2)
    fig_curves()
    R = rows()
    fig_alignment(R)
    fig_corners(R)
    for run in RUNS:
        fig_images(run)
        src = os.path.join(S1, f'fig5_{run}_mse200', f'fig5_maps_v15_s1_{run}_fig5_mse200_s0_ep100.png')
        shutil.copy(src, os.path.join(FIG, f's1_fig5_maps_{run}.png'))
    print('figures in', FIG)


if __name__ == '__main__':
    main()
