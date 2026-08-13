"""
v4 pre-R3 gate check (2026-08-14): does A-2's Q-value carry spatial
information when evaluated on the REAL R2-collected other_vision frames
(actual visited states, actual visitation density), as opposed to the
earlier synthetic-grid probe scan (probe_q_spatial_map.py) which used a
uniform grid A-2 never actually visits like that? This is the data R3/R4
will actually train on, so this is the relevant test for whether VE (R4)
has anything to learn from.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/check_a2_q_variance_real_data.py
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
N_SAMPLE = 8000

DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
A2_CRITIC = '/work/my_research/preference_inference/data/model/v3_rl_a2_critic.pth'
SAVE_DIR = 'data/result/baseline_v4'


def probe_actions(k):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def main():
    rng = np.random.RandomState(0)

    with h5py.File(DATA_H5, 'r') as f:
        n_data, seq_len = f['train/other_vision'].shape[:2]
        flat_n = n_data * seq_len
        idx = rng.choice(flat_n, size=min(N_SAMPLE, flat_n), replace=False)
        n_idx = idx // seq_len
        t_idx = idx % seq_len

        H, W, C = f['train/other_vision'].shape[2:]
        vision = np.zeros((len(idx), H, W, C), dtype=np.uint8)
        pos = np.zeros((len(idx), 2), dtype=np.float32)

        order = {}
        for i, (n, t) in enumerate(zip(n_idx, t_idx)):
            order.setdefault(int(n), []).append((i, int(t)))
        for n, items in order.items():
            ep_vision = f['train/other_vision'][n]
            ep_pos = f['train/other_position'][n]
            for i, t in items:
                vision[i] = ep_vision[t]
                pos[i] = ep_pos[t]

    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(A2_CRITIC, map_location=DEVICE))
    critic.eval()

    probes = torch.tensor(probe_actions(K_PROBES)).to(DEVICE)

    mean_q = np.zeros(len(vision))
    for i in range(len(vision)):
        v = vision[i]
        v_t = torch.tensor(v.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        v_t = v_t.repeat(K_PROBES, 1, 1, 1)
        with torch.no_grad():
            q, _ = critic(v_t, probes, hidden=None)
        mean_q[i] = q.squeeze(-1).cpu().numpy().mean()

    print(f'n_samples={len(vision)}')
    print(f'Q on REAL R2 A-2 data: range=[{mean_q.min():.4f}, {mean_q.max():.4f}]  '
          f'mean={mean_q.mean():.4f}  std={mean_q.std():.4f}  var={mean_q.var():.6f}')

    dist_to_green = np.linalg.norm(pos - np.array([-9.0, -9.0]), axis=1)
    corr_dist = float(np.corrcoef(mean_q, dist_to_green)[0, 1])
    print(f'corr(Q, distance_to_Green) = {corr_dist:.4f}  (expect negative: closer -> higher Q)')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].hist(mean_q, bins=60, color='seagreen')
    axes[0].set_title(f'A-2 Q distribution on real data (std={mean_q.std():.4f})')
    axes[0].set_xlabel('mean Q (8 probes)')

    sc = axes[1].scatter(pos[:, 0], pos[:, 1], c=mean_q, s=4, alpha=0.4, cmap='viridis')
    plt.colorbar(sc, ax=axes[1], label='mean Q')
    axes[1].set_title('A-2 Q colored by true position (real visited states)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_xlim(-10, 10)
    axes[1].set_ylim(-10, 10)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    plot_path = os.path.join(SAVE_DIR, 'a2_q_variance_real_data.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f'Saved: {plot_path}')

    result = {
        'n_samples': int(len(vision)),
        'q_range': [float(mean_q.min()), float(mean_q.max())],
        'q_mean': float(mean_q.mean()),
        'q_std': float(mean_q.std()),
        'q_var': float(mean_q.var()),
        'corr_q_dist_to_green': corr_dist,
        'note': 'measured on REAL R2-collected other_vision frames (actual visitation '
                'density), not a synthetic uniform grid scan',
    }
    out_path = os.path.join(SAVE_DIR, 'a2_q_variance_real_data.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
