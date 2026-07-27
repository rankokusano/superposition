"""
Compare VE's Q̂² estimate vs A-2's true Q² (approximated by Green proximity).

Since A-2 is trained with Green+1 reward, Q² is highest near Green (-9,-9).
We compare Q̂² with:
  1. Distance to Green (should correlate negatively)
  2. Scatter plot: Q̂² vs distance-to-Green per timestep
  3. Time series of Q̂² for several episodes

Usage (from /work/my_research/preference_inference):
    python analyze/plot_q2hat_v3.py
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

SAVED_H5  = 'data/result/v3_exp_b_mgve/0/test/v3_b_mgve_train/save/saved.h5'
SAVE_DIR  = 'data/result/v3_q2hat'
EPOCH     = 200
MODE      = 'eval'

GREEN_POS = np.array([-9.0, -9.0])
RED_POS   = np.array([-9.0,  9.0])
BLUE_POS  = np.array([ 9.0, -9.0])
CYAN_POS  = np.array([ 9.0,  9.0])


def load(h5_path, epoch, mode):
    with h5py.File(h5_path, 'r') as f:
        pre = f'{epoch:05d}/{mode}'
        q2_hat = f[f'{pre}/q2_hat/prediction'][()]        # (N, T, 1)
        op     = f[f'{pre}/other_position/input'][()]     # (N, T, 2)
        sp     = f[f'{pre}/self_position/input'][()]      # (N, T, 2)
    return q2_hat.squeeze(-1), op, sp   # (N,T), (N,T,2), (N,T,2)


def dist(pos, landmark):
    return np.linalg.norm(pos - landmark, axis=-1)  # (N, T)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f'Loading: {SAVED_H5}')
    q2_hat, op, sp = load(SAVED_H5, EPOCH, MODE)
    N, T = q2_hat.shape
    print(f'  q2_hat: {q2_hat.shape}  range [{q2_hat.min():.3f}, {q2_hat.max():.3f}]')

    d_green = dist(op, GREEN_POS)   # (N, T)
    d_red   = dist(op, RED_POS)
    d_blue  = dist(op, BLUE_POS)
    d_cyan  = dist(op, CYAN_POS)

    q_flat = q2_hat.flatten()
    dg_flat = d_green.flatten()

    # ── 1. Scatter: Q̂² vs distance to Green ─────────────────────────────────
    r, p = pearsonr(q_flat, -dg_flat)
    print(f'\nPearson r (Q̂² vs -dist_Green): {r:.4f}  p={p:.2e}')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(dg_flat, q_flat, gridsize=50, cmap='Blues', mincnt=1)
    ax.set_xlabel('Distance to Green landmark')
    ax.set_ylabel('Q̂² (VE output)')
    ax.set_title(f'Q̂² vs Distance to Green\nPearson r={r:.3f}')
    plt.colorbar(ax.collections[0], ax=ax, label='count')
    plt.tight_layout()
    p1 = os.path.join(SAVE_DIR, 'q2hat_vs_dist_green.png')
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f'Saved: {p1}')

    # ── 2. Q̂² mean by distance bin ──────────────────────────────────────────
    bins = np.linspace(0, dg_flat.max(), 20)
    bin_idx = np.digitize(dg_flat, bins)
    bin_mean_q = [q_flat[bin_idx == i].mean() if (bin_idx == i).sum() > 0 else np.nan
                  for i in range(1, len(bins))]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(bin_centers, bin_mean_q, 'o-', color='steelblue')
    ax.set_xlabel('Distance to Green landmark')
    ax.set_ylabel('Mean Q̂²')
    ax.set_title('Mean Q̂² by distance to Green\n(should decrease as distance increases)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(SAVE_DIR, 'q2hat_mean_by_dist.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f'Saved: {p2}')

    # ── 3. Time series: Q̂² and A-2 position for sample episodes ─────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle('Q̂² and A-2 distance to Green (sample episodes)')

    sample_eps = np.linspace(0, N - 1, 6, dtype=int)
    for idx, ep in enumerate(sample_eps):
        ax_q   = axes[idx // 2, (idx % 2)]
        t_axis = np.arange(T)

        color1, color2 = 'steelblue', 'green'
        ax_d = ax_q.twinx()

        ax_q.plot(t_axis, q2_hat[ep], color=color1, label='Q̂²', linewidth=1.2)
        ax_d.plot(t_axis, d_green[ep], color=color2, label='dist Green',
                  linestyle='--', linewidth=1.2, alpha=0.8)

        ax_q.set_xlabel('Step')
        ax_q.set_ylabel('Q̂²', color=color1)
        ax_d.set_ylabel('Distance to Green', color=color2)
        ax_q.set_title(f'Episode {ep}')
        ax_q.tick_params(axis='y', labelcolor=color1)
        ax_d.tick_params(axis='y', labelcolor=color2)

    plt.tight_layout()
    p3 = os.path.join(SAVE_DIR, 'q2hat_timeseries.png')
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f'Saved: {p3}')

    # ── 4. Q̂² stats summary ─────────────────────────────────────────────────
    summary = (
        f'=== Q̂² vs True Q² (proxy: Green proximity) ===\n'
        f'episodes x steps : {N} x {T}\n'
        f'Q̂² range         : [{q2_hat.min():.3f}, {q2_hat.max():.3f}]\n'
        f'Q̂² mean          : {q2_hat.mean():.3f}\n'
        f'dist_Green range  : [{d_green.min():.3f}, {d_green.max():.3f}]\n'
        f'Pearson r (Q̂² vs -dist_Green): {r:.4f}  (p={p:.2e})\n'
        f'\nInterpretation:\n'
        f'  r > 0 means Q̂² is higher when A-2 is closer to Green → VE learning OK\n'
        f'  r ≈ 0 means VE has not learned A-2 preference\n'
    )
    print(f'\n{summary}')
    txt = os.path.join(SAVE_DIR, 'q2hat_summary.txt')
    with open(txt, 'w') as f:
        f.write(summary)
    print(f'Saved: {txt}')


if __name__ == '__main__':
    main()
