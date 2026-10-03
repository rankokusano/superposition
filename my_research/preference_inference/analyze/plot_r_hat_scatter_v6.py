"""
S5 core figure (docs/v6_instructions.md Sec.10.3 "S5" row: "r_hat vs 真値の
散布図（論文の核心図）"). Scatters VE''s estimated r_hat (red vs green
components -- the only two nonzero components in either agent's true
reward vector) against A-1's and A-2's true reward points, colored by
A-2's distance to Green (the same bins used throughout Sec.27-38) to show
whether points drift toward A-2's true value at range.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/plot_r_hat_scatter_v6.py \
        --saved_h5 data/result/v6_s5_ve/0/test/r_hat_trend/save/saved.h5 \
        --epoch 400 --label v6_s5_ve
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = 'data/result/v6_baseline'
GREEN_POS = np.array([-9.0, -9.0])
A1_TRUE_R = np.array([1.0, -1.0, 0.0, 0.0])
A2_TRUE_R = np.array([-1.0, 1.0, 0.0, 0.0])
BIN_EDGES = [0.0, 5.0, 10.0, 15.0, 1e9]
BIN_LABELS = ['<5', '5-10', '10-15', '>=15']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saved_h5', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--mode', default='eval')
    ap.add_argument('--label', required=True)
    ap.add_argument('--n_sample', type=int, default=20000, help='subsample for plotting')
    args = ap.parse_args()

    with h5py.File(args.saved_h5, 'r') as f:
        g = f[f'{args.epoch:05d}/{args.mode}']
        r_hat = g['r_hat/prediction'][:]   # (N,T,4)
        op = g['other_position/truth'][:]  # (N,T,2)

    r_hat_flat = r_hat.reshape(-1, 4)
    op_flat = op.reshape(-1, 2)
    dist = np.linalg.norm(GREEN_POS[None, :] - op_flat, axis=1)

    rng = np.random.RandomState(0)
    n = len(r_hat_flat)
    idx = rng.choice(n, size=min(args.n_sample, n), replace=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap('viridis')
    bin_idx = np.digitize(dist[idx], BIN_EDGES[1:-1])
    sc = ax.scatter(r_hat_flat[idx, 0], r_hat_flat[idx, 1], c=bin_idx, cmap=cmap,
                     s=4, alpha=0.3, rasterized=True)
    cbar = plt.colorbar(sc, ax=ax, ticks=range(len(BIN_LABELS)))
    cbar.ax.set_yticklabels(BIN_LABELS)
    cbar.set_label('distance to Green')

    ax.scatter(*A1_TRUE_R[:2], marker='*', s=400, c='red', edgecolors='black',
               label='A-1 true r (+1,-1,0,0)', zorder=5)
    ax.scatter(*A2_TRUE_R[:2], marker='*', s=400, c='green', edgecolors='black',
               label='A-2 true r (-1,+1,0,0)', zorder=5)
    mean_r = r_hat_flat.mean(axis=0)
    ax.scatter(mean_r[0], mean_r[1], marker='X', s=200, c='orange', edgecolors='black',
               label='mean r_hat (all frames)', zorder=5)

    ax.set_xlabel('r_hat[red]')
    ax.set_ylabel('r_hat[green]')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f'{args.label} ep{args.epoch}: r_hat (red,green) vs true reward points')
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{args.label}_r_hat_scatter_ep{args.epoch}.png')
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
