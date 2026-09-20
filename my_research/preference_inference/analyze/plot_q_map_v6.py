"""
S1 gate figure (docs/v6_instructions.md §10.3): Q-value spatial maps for
the v6 reward-conditioned critic Q(s,a,r), one map per r condition, on a
shared color scale so the "value-range match" criterion (does Q have the
same order-of-magnitude spread for r=A1 vs r=A2, unlike v4/v5's separate
A1/A2 critics at 0.63 vs 0.009 std) is visible directly in the figure, not
just in printed std numbers.

r conditions plotted: A1's true r (+1,-1,0,0), A2's true r (-1,+1,0,0),
and an unseen r (0,+1,-1,0) never seen during training (continuous-uniform
sampling), per the 2026-09-13 addition to the S1 gate criteria
(docs/v6_experiment_log.md §2/§5).

Usage (inside Docker):
    cd /work/my_research/preference_inference
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python analyze/plot_q_map_v6.py --seed 0
"""
import argparse
import os
import sys

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import creator
from util import load_config
from my_research.preference_inference.model.rl_agent_sac_v6 import (
    ActorLSTM, CriticLSTM,
)

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DIR = '/work/my_research/preference_inference/data/model'
SAVE_DIR = '/work/my_research/preference_inference/data/result/v6_baseline'
ENV_CONFIG = '/work/simulation/config/collect/self_random_other_stay.yml'

GRID_N = 20
X_RANGE = np.linspace(-9.0, 9.0, GRID_N)
Y_RANGE = np.linspace(-9.0, 9.0, GRID_N)

LANDMARKS = {
    'Red': np.array([-9.0, 9.0]),
    'Green': np.array([-9.0, -9.0]),
    'Blue': np.array([9.0, -9.0]),
    'Cyan': np.array([9.0, 9.0]),
}
LANDMARK_COLORS = {'Red': 'red', 'Green': 'green', 'Blue': 'blue', 'Cyan': 'cyan'}

R_CONDITIONS = [
    ('A1_true (+1,-1,0,0)', np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)),
    ('A2_true (-1,+1,0,0)', np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32)),
    ('unseen (0,+1,-1,0)', np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32)),
]


def vision_to_tensor(v):
    v = v.astype(np.float32)
    v = np.transpose(v, (2, 0, 1))
    return torch.tensor(v).unsqueeze(0).to(DEVICE)


def r_to_tensor(w):
    return torch.tensor(np.asarray(w, dtype=np.float32)).unsqueeze(0).to(DEVICE)


def compute_q_grid(env, actor, critic, r_t):
    """Move self_agent on a grid (self camera), other_agent fixed at center."""
    q_grid = np.zeros((GRID_N, GRID_N))
    env.world.set_camera('self')
    env.other_agent.p = np.array([0.0, 0.0])

    for i, x in enumerate(X_RANGE):
        for j, y in enumerate(Y_RANGE):
            env.self_agent.p = np.array([x, y])
            env.world.draw(env.self_agent, env.other_agent)
            v = env.world.capture()
            v_t = vision_to_tensor(v)
            with torch.no_grad():
                a_t, _, _ = actor.sample(v_t, r_t, hidden=None)
                q, _ = critic(v_t, a_t, r_t, hidden=None)
            q_grid[i, j] = q.squeeze().cpu().item()
    return q_grid


def plot_q_map(q_grid, title, ax, vmin, vmax):
    im = ax.imshow(
        q_grid.T, origin='lower', cmap='hot', aspect='equal',
        extent=[-9, 9, -9, 9], vmin=vmin, vmax=vmax
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    for name, pos in LANDMARKS.items():
        ax.scatter(pos[0], pos[1], c=LANDMARK_COLORS[name], s=80, marker='D',
                   edgecolors='white', linewidths=0.5, zorder=5)
        ax.text(pos[0] + 0.4, pos[1] + 0.4, name, fontsize=8, color='white')
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default='',
                         help='matches train_rl_v6.py --tag, e.g. "relabel", '
                              'to load/save under that run\'s filenames')
    parser.add_argument('--film', action='store_true',
                         help='checkpoints were trained with --film')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    suffix = f'_{args.tag}' if args.tag else ''

    config = load_config(ENV_CONFIG)
    env = creator.create_environment(config.environment)
    env.init()
    env.off_display()

    actor = ActorLSTM(film=args.film).to(DEVICE)
    actor.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f'v6_rl_actor_seed{args.seed}{suffix}.pth'), map_location=DEVICE))
    actor.eval()
    critic = CriticLSTM(film=args.film).to(DEVICE)
    critic.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f'v6_rl_critic_seed{args.seed}{suffix}.pth'), map_location=DEVICE))
    critic.eval()

    grids = {}
    stats = {}
    for name, r_vec in R_CONDITIONS:
        print(f'Computing Q-map for r = {name} ({GRID_N}x{GRID_N})...')
        r_t = r_to_tensor(r_vec)
        q_grid = compute_q_grid(env, actor, critic, r_t)
        grids[name] = q_grid
        stats[name] = dict(min=float(q_grid.min()), max=float(q_grid.max()), std=float(q_grid.std()))
        print(f'  r={name}: min={stats[name]["min"]:.4f} max={stats[name]["max"]:.4f} std={stats[name]["std"]:.4f}')

    # shared color scale across A1/A2 (the point of the figure: same-order value range)
    a1_a2_all = np.concatenate([grids[R_CONDITIONS[0][0]].ravel(), grids[R_CONDITIONS[1][0]].ravel()])
    vmin, vmax = float(a1_a2_all.min()), float(a1_a2_all.max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (name, _) in zip(axes, R_CONDITIONS):
        s = stats[name]
        im = plot_q_map(grids[name], f'r = {name}\nstd={s["std"]:.4f}', ax, vmin, vmax)
        plt.colorbar(im, ax=ax, label='Q-value')

    std_ratio = max(stats[R_CONDITIONS[0][0]]['std'], stats[R_CONDITIONS[1][0]]['std']) / \
        max(min(stats[R_CONDITIONS[0][0]]['std'], stats[R_CONDITIONS[1][0]]['std']), 1e-8)
    plt.suptitle(
        f'v6 S1: conditioned critic Q(s,a,r) spatial maps (seed={args.seed}), '
        f'shared color scale on A1/A2 panels, std ratio={std_ratio:.2f}x',
        fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, f'q_map_v6_seed{args.seed}{suffix}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')

    # individual panels too
    for name, _ in R_CONDITIONS:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        s = stats[name]
        im = plot_q_map(grids[name], f'r = {name}\nstd={s["std"]:.4f}', ax, vmin, vmax)
        plt.colorbar(im, ax=ax, label='Q-value')
        plt.tight_layout()
        safe_name = name.split(' ')[0]
        p = os.path.join(SAVE_DIR, f'q_map_v6_seed{args.seed}{suffix}_{safe_name}.png')
        plt.savefig(p, dpi=150)
        plt.close()
        print(f'Saved: {p}')

    import json
    stats_path = os.path.join(SAVE_DIR, f'q_map_v6_seed{args.seed}{suffix}_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Saved: {stats_path}')

    # raw grids too, for quantitative cross-run comparison (e.g. corr(Q_A1, -Q_A2))
    grids_path = os.path.join(SAVE_DIR, f'q_map_v6_seed{args.seed}{suffix}_grids.npz')
    np.savez(grids_path, **{name.split(' ')[0]: grids[name] for name, _ in R_CONDITIONS})
    print(f'Saved: {grids_path}')


if __name__ == '__main__':
    main()
