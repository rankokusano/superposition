"""
Plot mean Q̂² at each A-2 position, averaged over actual episode data.

For every timestep in every episode, bin A-2's position and accumulate Q̂².
Then plot the mean Q̂² per bin as a heatmap.

This reflects what VE actually outputs during real episodes
(no fixed A-1 position assumption).

Usage (from /work/my_research/preference_inference inside Docker):
    python analyze/plot_q2hat_by_a2pos.py
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVED_H5 = 'data/result/v3_exp_b_mgve/0/test/v3_b_mgve_train/save/saved.h5'
SAVE_DIR = 'data/result/v3_q2hat'
EPOCH    = 200
MODE     = 'eval'
GRID_N   = 25
ARENA    = (-10, 10)

LANDMARKS = {
    'Red':   np.array([-9.0,  9.0]),
    'Green': np.array([-9.0, -9.0]),
    'Blue':  np.array([ 9.0, -9.0]),
    'Cyan':  np.array([ 9.0,  9.0]),
}
LM_COLORS = {'Red': 'red', 'Green': 'green', 'Blue': 'blue', 'Cyan': 'cyan'}


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    with h5py.File(SAVED_H5, 'r') as f:
        pre = f'{EPOCH:05d}/{MODE}'
        q2_hat = f[f'{pre}/q2_hat/prediction'][()]    # (N, T, 1)
        op     = f[f'{pre}/other_position/input'][()]  # (N, T, 2)

    q2_hat = q2_hat.squeeze(-1)  # (N, T)
    N, T = q2_hat.shape

    # flatten
    q_flat = q2_hat.flatten()          # (N*T,)
    ox_flat = op[:, :, 0].flatten()    # (N*T,)
    oy_flat = op[:, :, 1].flatten()    # (N*T,)

    # bin edges
    edges = np.linspace(ARENA[0], ARENA[1], GRID_N + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 2D accumulation
    q_sum   = np.zeros((GRID_N, GRID_N))
    q_count = np.zeros((GRID_N, GRID_N))

    ix = np.digitize(ox_flat, edges) - 1
    iy = np.digitize(oy_flat, edges) - 1
    mask = (ix >= 0) & (ix < GRID_N) & (iy >= 0) & (iy < GRID_N)

    np.add.at(q_sum,   (ix[mask], iy[mask]), q_flat[mask])
    np.add.at(q_count, (ix[mask], iy[mask]), 1)

    q_mean = np.where(q_count > 0, q_sum / q_count, np.nan)

    valid = q_count > 0
    print(f'Covered bins: {valid.sum()}/{GRID_N**2}')
    print(f'Q̂² mean range: [{np.nanmin(q_mean):.3f}, {np.nanmax(q_mean):.3f}]')

    # ── main heatmap ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        q_mean.T, origin='lower', cmap='hot', aspect='equal',
        extent=[ARENA[0], ARENA[1], ARENA[0], ARENA[1]],
        vmin=np.nanpercentile(q_mean, 5),
        vmax=np.nanpercentile(q_mean, 95),
    )
    plt.colorbar(im, ax=ax, label='Mean Q̂² (VE output)')
    ax.set_title('Mean Q̂² by A-2 position\n(averaged over actual episode data)')
    ax.set_xlabel('A-2 position x')
    ax.set_ylabel('A-2 position y')

    for name, pos in LANDMARKS.items():
        ax.scatter(pos[0], pos[1], c=LM_COLORS[name], s=100, marker='D',
                   edgecolors='white', linewidths=0.8, zorder=5)
        ax.text(pos[0] + 0.4, pos[1] + 0.4, name, fontsize=9,
                color='white', fontweight='bold')

    plt.tight_layout()
    p1 = os.path.join(SAVE_DIR, 'q2hat_mean_by_a2pos_actual.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f'Saved: {p1}')

    # ── visit count heatmap（参考） ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    im2 = ax.imshow(
        q_count.T, origin='lower', cmap='Blues', aspect='equal',
        extent=[ARENA[0], ARENA[1], ARENA[0], ARENA[1]],
    )
    plt.colorbar(im2, ax=ax, label='Visit count')
    ax.set_title('A-2 position visit count')
    ax.set_xlabel('A-2 position x')
    ax.set_ylabel('A-2 position y')
    for name, pos in LANDMARKS.items():
        ax.scatter(pos[0], pos[1], c=LM_COLORS[name], s=80, marker='D',
                   edgecolors='white', linewidths=0.8, zorder=5)
        ax.text(pos[0] + 0.4, pos[1] + 0.4, name, fontsize=8, color='white')
    plt.tight_layout()
    p2 = os.path.join(SAVE_DIR, 'a2_visit_count.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f'Saved: {p2}')


if __name__ == '__main__':
    main()
