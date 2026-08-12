"""
v4 Sec.5.3-style probe-Q spatial map, used here as a diagnostic (not yet
wired into the SM input -- that's R3+). For A-1's and A-2's critics
separately: place the agent at each point of a grid spanning the arena,
capture its vision, evaluate the critic at K fixed directional probe
actions, and report the spatial range/variance of the resulting Q.

This is the direct check for whether removing the reward threshold (Sec 5.2
revision) actually restored spatial information to Q: a constant Q across
the grid would mean R4 (VE predicting true Q2) and R5 (self-projection
heatmap) can't be meaningful, no matter how good the representation is
otherwise.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/probe_q_spatial_map.py   # default: canonical a1/a2 paths
    python analyze/probe_q_spatial_map.py --critic_path data/model/v3_rl_critic_seed0.pth \
        --env_config /work/simulation/config/collect/self_random_other_stay.yml \
        --camera self --scan_self --label a1_seed0
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import creator  # noqa
from util import load_config  # noqa
from my_research.rl_agent_sac import CriticLSTM  # noqa

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRID_N = 10  # matches the 10x10 text-grid used elsewhere in this analysis
K_PROBES = 8
COUNTERPART_POS = np.array([0.0, 0.0])  # fixed position of the non-scanned agent

LANDMARKS = {
    'Red':    np.array([-9.0,  9.0]),
    'Green':  np.array([-9.0, -9.0]),
    'Blue':   np.array([ 9.0, -9.0]),
    'Yellow': np.array([ 9.0,  9.0]),
}

SAVE_DIR = 'data/result/baseline_v4'


def vision_to_tensor(v):
    v = v.astype(np.float32)
    return torch.tensor(np.transpose(v, (2, 0, 1))).unsqueeze(0).to(DEVICE)


def probe_actions(k):
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)


def grid_centers(n):
    edges = np.linspace(-9.5, 9.5, n + 1)
    return (edges[:-1] + edges[1:]) / 2


def scan(critic_path, env_config, scan_self, camera_key, label):
    env_cfg = load_config(env_config)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()
    env.reset()
    # fix the counterpart at a constant position for a fair scan
    if scan_self:
        env.other_agent.p = COUNTERPART_POS.copy()
    else:
        env.self_agent.p = COUNTERPART_POS.copy()

    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))
    critic.eval()

    probes = torch.tensor(probe_actions(K_PROBES)).to(DEVICE)  # (K, 2)

    centers = grid_centers(GRID_N)
    mean_q = np.zeros((GRID_N, GRID_N))
    std_q = np.zeros((GRID_N, GRID_N))

    for ix, x in enumerate(centers):
        for iy, y in enumerate(centers):
            agent = env.self_agent if scan_self else env.other_agent
            agent.p = np.array([x, y])
            env.world.set_camera(camera_key)
            env.world.draw(env.self_agent, env.other_agent)
            v = env.world.capture()
            v_t = vision_to_tensor(v).repeat(K_PROBES, 1, 1, 1)
            with torch.no_grad():
                q, _ = critic(v_t, probes, hidden=None)
            q_np = q.squeeze(-1).cpu().numpy()
            mean_q[ix, iy] = q_np.mean()
            std_q[ix, iy] = q_np.std()

    print(f'\n=== {label}: probe-Q spatial map (grid {GRID_N}x{GRID_N}, K={K_PROBES} probes) ===')
    print(f'mean_q range: [{mean_q.min():.4f}, {mean_q.max():.4f}]  '
          f'spatial variance of mean_q: {mean_q.var():.6f}  spatial std: {mean_q.std():.4f}')
    print(f'within-point std across probes (avg over grid): {std_q.mean():.4f}')

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(mean_q.T, origin='lower', extent=[-9.5, 9.5, -9.5, 9.5], cmap='coolwarm')
    plt.colorbar(im, ax=ax, label='mean Q over probes')
    for name, lp in LANDMARKS.items():
        ax.plot(lp[0], lp[1], 'k+', markersize=14, markeredgewidth=2)
        ax.annotate(name, lp, color='black', fontsize=8, ha='center', va='bottom')
    ax.set_title(f'{label}: probe-Q spatial map')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f'{label}_q_spatial_map.png')
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'Saved: {save_path}')

    print('  text grid (mean Q, rows=y high->low, cols=x low->high):')
    grid_display = mean_q.T[::-1]
    for row in grid_display:
        print('   ', ' '.join(f'{v:6.3f}' for v in row))

    return mean_q, std_q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic_path')
    parser.add_argument('--env_config')
    parser.add_argument('--camera', choices=['self', 'other'])
    parser.add_argument('--scan_self', action='store_true')
    parser.add_argument('--label')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    if args.critic_path:
        scan(args.critic_path, args.env_config, scan_self=args.scan_self,
             camera_key=args.camera, label=args.label)
        return

    # no args: default behaviour, scan both a1 and a2 with the canonical paths
    a1_critic = '/work/my_research/preference_inference/data/model/v3_rl_critic.pth'
    a2_critic = '/work/my_research/preference_inference/data/model/v3_rl_a2_critic.pth'

    scan(a1_critic, '/work/simulation/config/collect/self_random_other_stay.yml',
         scan_self=True, camera_key='self', label='a1')
    scan(a2_critic, '/work/simulation/config/collect/self_stay_other_random.yml',
         scan_self=False, camera_key='other', label='a2')


if __name__ == '__main__':
    main()
