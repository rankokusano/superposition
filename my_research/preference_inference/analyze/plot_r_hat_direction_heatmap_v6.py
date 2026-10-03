"""
S5 required figure (docs/v6_instructions.md Sec.10.3 "S5": direction heatmap).
Adapts r4_ve_eval.py's Item-4 heatmap: for each A-2 position bin, the mean
cos_sim between Q_hat2's peak probe-direction and the TRUE direction to
Green, to see spatially where (if anywhere) the direction signal is correct.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/plot_r_hat_direction_heatmap_v6.py \
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
K = 8
PROBE_ANGLES = np.array([2 * np.pi * i / K for i in range(K)])
PROBE_ACTIONS = np.stack([np.cos(PROBE_ANGLES), np.sin(PROBE_ANGLES)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saved_h5', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--mode', default='eval')
    ap.add_argument('--label', required=True)
    ap.add_argument('--bins', type=int, default=15)
    args = ap.parse_args()

    with h5py.File(args.saved_h5, 'r') as f:
        g = f[f'{args.epoch:05d}/{args.mode}']
        q2_hat = g['q2_hat/prediction'][:]
        op = g['other_position/truth'][:]

    q2_flat = q2_hat.reshape(-1, K)
    op_flat = op.reshape(-1, 2)

    to_green = GREEN_POS[None, :] - op_flat
    to_green_norm = to_green / (np.linalg.norm(to_green, axis=1, keepdims=True) + 1e-8)
    argmax_idx = q2_flat.argmax(axis=1)
    argmax_dir = PROBE_ACTIONS[argmax_idx]
    cos_sim = (argmax_dir * to_green_norm).sum(axis=1)

    bins = args.bins
    xedges = np.linspace(-9.5, 9.5, bins + 1)
    yedges = np.linspace(-9.5, 9.5, bins + 1)
    xi = np.clip(np.digitize(op_flat[:, 0], xedges) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(op_flat[:, 1], yedges) - 1, 0, bins - 1)
    heat_sum = np.zeros((bins, bins))
    heat_cnt = np.zeros((bins, bins))
    for i in range(len(cos_sim)):
        heat_sum[yi[i], xi[i]] += cos_sim[i]
        heat_cnt[yi[i], xi[i]] += 1
    heat = np.divide(heat_sum, heat_cnt, out=np.full_like(heat_sum, np.nan), where=heat_cnt > 0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(heat, origin='lower', extent=[-9.5, 9.5, -9.5, 9.5], cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='cos_sim(Q_hat2 argmax dir, true dir to Green)')
    ax.plot(-9, -9, 'g*', markersize=15, label='Green')
    ax.plot(-9, 9, 'r*', markersize=15, label='Red')
    ax.plot(9, -9, 'b*', markersize=15, label='Blue')
    ax.plot(9, 9, 'c*', markersize=15, label='Cyan')
    ax.set_xlabel('A-2 x'); ax.set_ylabel('A-2 y')
    ax.set_title(f"{args.label} ep{args.epoch}: Q_hat2 peak-direction alignment with Green, by A-2 pos")
    ax.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{args.label}_direction_heatmap_ep{args.epoch}.png')
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
