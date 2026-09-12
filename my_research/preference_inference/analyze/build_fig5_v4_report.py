"""
Build a compact, advisor-facing summary figure for the Fig.5-v4 (paper-faithful
viewpoint-taking decoder) analysis.

Since the Visual Encoder is frozen identically across all v4 stages (verified
via torch.equal on checkpoint weights AND via byte-identical analyze_vpt.py
outputs -- see docs/v4_experiment_log.md SS8.5), exp1_l1_1000 and r3_stay_400
produce pixel-identical Fig.5 results. This figure therefore represents BOTH
models jointly -- a separate r3_stay_400 copy is intentionally not produced,
since the identity itself (not two independent-looking measurements) is the
finding.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/build_fig5_v4_report.py
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAP_DIR = 'data/result/fig5_v4_exp1l1000/0/test/grid/save/vpt/1/map'
OUT_PATH = 'data/result/fig5_v4_exp1l1000/fig5_v4_common.png'

COMBOS = [
    ('0', '0', '0', '0'),
    ('0', '0', '10', '10'),
    ('10', '10', '0', '0'),
    ('10', '10', '10', '10'),
    ('0', '10', '10', '0'),
]

COLS = ['self_true', 'self_rec', 'other_true', 'other_rec']
COL_TITLES = ['A-1 real vision', 'Encoder-1 reconstruction',
              'A-2 real vision', 'Encoder-2 reconstruction']


def load_bgr_to_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    fig, axes = plt.subplots(len(COMBOS), 4, figsize=(12, 2.3 * len(COMBOS) + 0.9))
    fig.suptitle(
        'Fig.5-style viewpoint-taking decoder -- shared by exp1_l1_1000 AND r3_stay_400 (epoch 10)\n'
        'Visual Encoder is frozen at every v4 stage; weights are byte-identical between the two models '
        '(verified via np.array_equal). This single figure represents BOTH.',
        fontsize=10, y=0.99)

    for row, (sx, sy, ox, oy) in enumerate(COMBOS):
        prefix = f'other_ae_self_{sx}_{sy}_other_{ox}_{oy}_'
        for col, key in enumerate(COLS):
            path = os.path.join(MAP_DIR, prefix + key + '.png')
            img = load_bgr_to_rgb(path)
            axes[row, col].imshow(img)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(COL_TITLES[col], fontsize=10)
        axes[row, 0].set_ylabel(f'self=({sx},{sy})\nother=({ox},{oy})', fontsize=8)

    plt.subplots_adjust(top=0.90, bottom=0.02, left=0.06, right=0.99, hspace=0.15, wspace=0.05)
    plt.savefig(OUT_PATH, dpi=150)
    print(f'Saved: {OUT_PATH}')


if __name__ == '__main__':
    main()
