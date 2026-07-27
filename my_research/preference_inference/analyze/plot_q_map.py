"""
Plot Q-value heatmap from grid data.

For each A-2 position on the grid, show the Q-value Q^1_t as a heatmap
over A-1 positions.  This visualizes whether A-1's expected return depends
on A-2's position (and hence A-2's inferred preference).

Usage (inside Docker /work):
    cd /work/my_research/preference_inference/analyze
    python plot_q_map.py \
        --h5 /work/my_research/preference_inference/data/data/new_grid_q_map/data.h5 \
        --savedir /work/my_research/preference_inference/data/result/q_map \
        --num_div 20
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LANDMARK_POSITIONS = {
    'Red':    np.array([-9.0,  9.0]),
    'Green':  np.array([-9.0, -9.0]),
    'Blue':   np.array([ 9.0, -9.0]),
    'Yellow': np.array([ 9.0,  9.0]),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', required=True, help='grid data h5 file')
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--num_div', type=int, default=20,
                        help='number of grid divisions per axis')
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)
    nd = args.num_div

    with h5py.File(args.h5, 'r') as f:
        # shape: (N_other, N_self, ...) where N_other = nd*nd, N_self = nd*nd
        self_pos   = f[f'{args.mode}/self_position'][()]    # (N_o, N_s, 2)
        other_pos  = f[f'{args.mode}/other_position'][()]   # (N_o, N_s, 2)
        q_values   = f[f'{args.mode}/a1_q_values'][()]      # (N_o, N_s, 1)

    N_o = nd * nd
    N_s = nd * nd

    # Reshape to (nd, nd, nd, nd, ...)
    # outer grid = A-2 (ox, oy), inner grid = A-1 (sx, sy)
    q_map = q_values.reshape(nd, nd, nd, nd)      # (ox, oy, sx, sy)
    sp_map = self_pos.reshape(nd, nd, nd, nd, 2)  # for axis labels
    op_map = other_pos.reshape(nd, nd, nd, nd, 2)

    # Get axis values from the first row/col
    sx_vals = sp_map[0, 0, :, 0, 0]  # sx values
    sy_vals = sp_map[0, 0, 0, :, 1]  # sy values

    # Plot Q-value heatmap for a few A-2 positions
    # Select 4 A-2 positions near each landmark
    xmin, xmax = -10, 10
    def pos_to_idx(x, n):
        return int(np.clip(np.round((x - xmin) / (xmax - xmin) * (n - 1)), 0, n - 1))

    landmark_idxs = {
        name: (pos_to_idx(pos[0], nd), pos_to_idx(pos[1], nd))
        for name, pos in LANDMARK_POSITIONS.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, (lm_name, (oi, oj)) in zip(axes, landmark_idxs.items()):
        q = q_map[oi, oj]  # (nd, nd) Q-values over A-1 positions
        im = ax.imshow(q.T, origin='lower', cmap='hot',
                       extent=[xmin, xmax, xmin, xmax], aspect='equal')
        ax.set_title(f'A-2 near {lm_name}')
        ax.set_xlabel('A-1 sx')
        ax.set_ylabel('A-1 sy')

        # Mark A-2 actual position
        ox_val = op_map[oi, oj, 0, 0, 0]
        oy_val = op_map[oi, oj, 0, 0, 1]
        ax.scatter([ox_val], [oy_val], c='cyan', s=100, marker='*',
                   label=f'A-2 ({ox_val:.1f},{oy_val:.1f})')

        # Mark landmarks
        for ln, lp in LANDMARK_POSITIONS.items():
            ax.scatter(lp[0], lp[1], c='blue', s=60, marker='D')
            ax.text(lp[0] + 0.3, lp[1] + 0.3, ln, fontsize=7, color='blue')

        plt.colorbar(im, ax=ax, label='Q¹_t')

    plt.tight_layout()
    save_path = os.path.join(args.savedir, 'q_map_by_a2_position.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

    # Also save a summary heatmap: mean Q-value vs A-2 position
    # Flatten over A-1 positions
    q_mean_by_a2 = q_map.mean(axis=(2, 3))  # (nd, nd)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(q_mean_by_a2.T, origin='lower', cmap='hot',
                   extent=[xmin, xmax, xmin, xmax], aspect='equal')
    plt.colorbar(im, ax=ax, label='mean Q¹_t over A-1 positions')
    ax.set_title('Mean A-1 Q-value as function of A-2 position')
    ax.set_xlabel('A-2 position x')
    ax.set_ylabel('A-2 position y')
    for ln, lp in LANDMARK_POSITIONS.items():
        ax.scatter(lp[0], lp[1], c='cyan', s=80, marker='D')
        ax.text(lp[0] + 0.3, lp[1] + 0.3, ln, fontsize=8, color='cyan')
    plt.tight_layout()
    summary_path = os.path.join(args.savedir, 'q_mean_by_a2.png')
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"Saved: {summary_path}")
