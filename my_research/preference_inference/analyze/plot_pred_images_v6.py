"""
S2 prediction-image figure (docs/v6_instructions.md §10.1/10.3): a montage
of self_vision {truth, prediction} and other_vision {truth} across a few
sampled timesteps, from the viz20 saved.h5 (small dataset, kept for exactly
this purpose per docs/v6_experiment_log.md §4.3/29.1).

Usage (no GPU needed, just h5py + matplotlib -- run inside Docker):
    python analyze/plot_pred_images_v6.py \
        --saved_h5 data/result/v6_s2_base_mse/0/test/viz20/save/saved.h5 \
        --epoch 400 --label v6_s2_base_mse
"""
import argparse
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SAVE_DIR = 'data/result/v6_baseline'
N_EPISODES = 4
STEPS = [10, 40, 70, 99]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saved_h5', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--mode', default='eval')
    args = ap.parse_args()

    with h5py.File(args.saved_h5, 'r') as f:
        g = f[f'{args.epoch:05d}/{args.mode}']
        sv_truth = g['self_vision/truth'][:]
        sv_pred = g['self_vision/prediction'][:]
        ov_truth = g['other_vision/truth'][:]

    n_ep = min(N_EPISODES, sv_truth.shape[0])
    rows = 3 * n_ep  # self truth, self prediction, other truth, per episode
    cols = len(STEPS)
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    row_labels = ['self truth', 'self prediction', 'other truth']

    def show(ax, img):
        img = np.clip((img + 1) / 2, 0, 1) if img.min() < 0 else np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

    for e in range(n_ep):
        for j, t in enumerate(STEPS):
            r0 = 3 * e
            show(axes[r0, j], sv_truth[e, t])
            show(axes[r0 + 1, j], sv_pred[e, t])
            show(axes[r0 + 2, j], ov_truth[e, t])
            if e == 0:
                axes[r0, j].set_title(f't={t}', fontsize=10)
        for k in range(3):
            axes[3 * e + k, 0].set_ylabel(f'ep{e} {row_labels[k]}', fontsize=8)

    plt.suptitle(f'{args.label} ep{args.epoch} ({args.mode}): self truth / self prediction / other truth', fontsize=12)
    plt.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{args.label}_pred_images_ep{args.epoch}.png')
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
