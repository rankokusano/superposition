"""
v4 pre-R3 check (2026-08-14): does A-1's Q-value actually correlate with
red/green pixel count in the same state? The professor's note ("Q high ->
remember red, Q low -> remember green") is a value<->visual-feature
correspondence requirement for VE (R4) to be feasible at all -- if Q
doesn't track the visual signal the reward was literally defined from,
VE has nothing to learn from vision.

Uses A-1's canonical critic (data/model/v3_rl_critic.pth, seed0) and A-1's
own real collected vision frames (self_vision from r2_a1random_a2rl/data.h5,
A-1 moves via RandomAgent there so it's a good spatial sample), evaluates
Q(v, a_probe) for K=8 directional probes averaged per-frame, and correlates
against the actual red/green pixel counts in that same frame.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/check_q_vision_correlation.py
"""
import json
import os
import sys

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/work')
from my_research.rl_agent_sac import CriticLSTM  # noqa

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_PROBES = 8
N_SAMPLE = 8000  # subsample for speed

DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
A1_CRITIC = '/work/my_research/preference_inference/data/model/v3_rl_critic.pth'
SAVE_DIR = 'data/result/baseline_v4'


def probe_actions(k):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def pixel_counts(vision_uint8):
    v = vision_uint8.astype(np.float32) / 255.0
    r, g, b = v[:, :, 0], v[:, :, 1], v[:, :, 2]
    red = int(((r > 0.9) & (g < 0.1) & (b < 0.1)).sum())
    green = int(((g > 0.9) & (r < 0.1) & (b < 0.1)).sum())
    return red, green


def main():
    rng = np.random.RandomState(0)

    with h5py.File(DATA_H5, 'r') as f:
        n_data, seq_len = f['train/self_vision'].shape[:2]
        flat_n = n_data * seq_len
        idx = rng.choice(flat_n, size=min(N_SAMPLE, flat_n), replace=False)
        idx = np.sort(idx)
        n_idx = idx // seq_len
        t_idx = idx % seq_len

        # group by episode to read contiguous slices efficiently
        vision = np.zeros((len(idx), seq_len and f['train/self_vision'].shape[2],
                            f['train/self_vision'].shape[3], f['train/self_vision'].shape[4]), dtype=np.uint8)
        order = {}
        for i, (n, t) in enumerate(zip(n_idx, t_idx)):
            order.setdefault(int(n), []).append((i, int(t)))
        for n, items in order.items():
            ep_vision = f['train/self_vision'][n]  # (seq_len, H, W, 3) uint8
            for i, t in items:
                vision[i] = ep_vision[t]

    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(A1_CRITIC, map_location=DEVICE))
    critic.eval()

    probes = torch.tensor(probe_actions(K_PROBES)).to(DEVICE)

    mean_q = np.zeros(len(vision))
    red_counts = np.zeros(len(vision))
    green_counts = np.zeros(len(vision))

    for i in range(len(vision)):
        v = vision[i]
        red, green = pixel_counts(v)
        red_counts[i] = red
        green_counts[i] = green

        v_t = torch.tensor(v.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        v_t = v_t.repeat(K_PROBES, 1, 1, 1)
        with torch.no_grad():
            q, _ = critic(v_t, probes, hidden=None)
        mean_q[i] = q.squeeze(-1).cpu().numpy().mean()

    red_frac = red_counts / (vision.shape[1] * vision.shape[2])
    green_frac = green_counts / (vision.shape[1] * vision.shape[2])

    corr_red = float(np.corrcoef(mean_q, red_frac)[0, 1])
    corr_green = float(np.corrcoef(mean_q, green_frac)[0, 1])
    corr_reward_proxy = float(np.corrcoef(mean_q, red_frac - green_frac)[0, 1])

    print(f'n_samples={len(vision)}')
    print(f'corr(Q, red_fraction)   = {corr_red:.4f}')
    print(f'corr(Q, green_fraction) = {corr_green:.4f}')
    print(f'corr(Q, red_fraction - green_fraction) = {corr_reward_proxy:.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(red_frac, mean_q, s=2, alpha=0.15, color='crimson', label='red_fraction')
    axes[0].scatter(green_frac, mean_q, s=2, alpha=0.15, color='seagreen', label='green_fraction')
    axes[0].set_xlabel('pixel fraction')
    axes[0].set_ylabel('mean Q (8 probes)')
    axes[0].legend()
    axes[0].set_title('Q vs red/green pixel fraction')

    axes[1].scatter(red_frac - green_frac, mean_q, s=2, alpha=0.15, color='purple')
    axes[1].set_xlabel('red_fraction - green_fraction (reward proxy)')
    axes[1].set_ylabel('mean Q (8 probes)')
    axes[1].set_title(f'Q vs reward-proxy (corr={corr_reward_proxy:.3f})')

    plt.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    plot_path = os.path.join(SAVE_DIR, 'a1_q_vs_vision_correlation.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f'Saved: {plot_path}')

    result = {
        'n_samples': int(len(vision)),
        'corr_q_red_fraction': corr_red,
        'corr_q_green_fraction': corr_green,
        'corr_q_reward_proxy': corr_reward_proxy,
    }
    out_path = os.path.join(SAVE_DIR, 'a1_q_vision_correlation.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
