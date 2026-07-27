"""
Plot Q-value heatmaps for A-1 (v3) and A-2 (v3) side by side.

Places each agent on a grid, captures vision, computes Q-value using
the respective RL critic. The other agent is fixed at the arena center.

Usage (inside Docker):
    cd /work/my_research/preference_inference
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python analyze/plot_q_map_v3.py
"""

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
from my_research.rl_agent_sac import ActorLSTM, CriticLSTM

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

A1_CRITIC_PATH = '/work/my_research/preference_inference/data/model/v3_rl_critic.pth'
A2_CRITIC_PATH = '/work/my_research/preference_inference/data/model/v3_rl_a2_critic.pth'
A1_ACTOR_PATH  = '/work/my_research/preference_inference/data/model/v3_rl_actor.pth'
A2_ACTOR_PATH  = '/work/my_research/preference_inference/data/model/v3_rl_a2_actor.pth'

SAVE_DIR = '/work/my_research/preference_inference/data/result/q_map_v3'
ENV_CONFIG = '/work/simulation/config/collect/self_random_other_stay.yml'

GRID_N   = 20
X_RANGE  = np.linspace(-9.0, 9.0, GRID_N)
Y_RANGE  = np.linspace(-9.0, 9.0, GRID_N)

LANDMARKS = {
    'Red':   np.array([-9.0,  9.0]),
    'Green': np.array([-9.0, -9.0]),
    'Blue':  np.array([ 9.0, -9.0]),
    'Cyan':  np.array([ 9.0,  9.0]),
}
LANDMARK_COLORS = {'Red': 'red', 'Green': 'green', 'Blue': 'blue', 'Cyan': 'cyan'}


def load_agent(actor_path, critic_path):
    actor = ActorLSTM().to(DEVICE)
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    actor.eval()
    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(critic_path, map_location=DEVICE))
    critic.eval()
    return actor, critic


def vision_to_tensor(v):
    v = v.astype(np.float32)
    v = np.transpose(v, (2, 0, 1))
    return torch.tensor(v).unsqueeze(0).to(DEVICE)


def compute_q_grid(env, actor, critic, use_self=True):
    """
    Compute Q-value at each grid position for the target agent.
    use_self=True: move self_agent on grid, other_agent fixed at center
    use_self=False: move other_agent on grid, self_agent fixed at center (A-2 camera)
    """
    q_grid = np.zeros((GRID_N, GRID_N))
    center = np.array([0.0, 0.0])

    if use_self:
        env.world.set_camera('self')
        env.other_agent.p = center.copy()
    else:
        env.world.set_camera('other')
        env.self_agent.p = center.copy()

    for i, x in enumerate(X_RANGE):
        for j, y in enumerate(Y_RANGE):
            pos = np.array([x, y])
            if use_self:
                env.self_agent.p = pos.copy()
            else:
                env.other_agent.p = pos.copy()

            env.world.draw(env.self_agent, env.other_agent)
            v = env.world.capture()
            v_t = vision_to_tensor(v)

            with torch.no_grad():
                a_t, _, _ = actor.sample(v_t, hidden=None)
                q1, _ = critic(v_t, a_t, hidden=None)
            q_grid[i, j] = q1.squeeze().cpu().item()

    return q_grid


def plot_q_map(q_grid, title, ax, vmin=None, vmax=None):
    im = ax.imshow(
        q_grid.T, origin='lower', cmap='hot', aspect='equal',
        extent=[-9, 9, -9, 9], vmin=vmin, vmax=vmax
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    for name, pos in LANDMARKS.items():
        ax.scatter(pos[0], pos[1], c=LANDMARK_COLORS[name], s=80, marker='D',
                   edgecolors='white', linewidths=0.5, zorder=5)
        ax.text(pos[0] + 0.4, pos[1] + 0.4, name, fontsize=8, color='white')
    return im


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    config = load_config(ENV_CONFIG)
    env = creator.create_environment(config.environment)
    env.init()
    env.off_display()

    print('Loading A-1 agents...')
    a1_actor, a1_critic = load_agent(A1_ACTOR_PATH, A1_CRITIC_PATH)
    print('Loading A-2 agents...')
    a2_actor, a2_critic = load_agent(A2_ACTOR_PATH, A2_CRITIC_PATH)

    print(f'Computing A-1 Q-map ({GRID_N}x{GRID_N})...')
    q_a1 = compute_q_grid(env, a1_actor, a1_critic, use_self=True)
    print(f'  A-1 Q range: {q_a1.min():.2f} ~ {q_a1.max():.2f}')

    print(f'Computing A-2 Q-map ({GRID_N}x{GRID_N})...')
    q_a2 = compute_q_grid(env, a2_actor, a2_critic, use_self=False)
    print(f'  A-2 Q range: {q_a2.min():.2f} ~ {q_a2.max():.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im1 = plot_q_map(q_a1, 'A-1 Q-value map\n(Red+1, Green-1)', axes[0])
    plt.colorbar(im1, ax=axes[0], label='Q-value')

    im2 = plot_q_map(q_a2, 'A-2 Q-value map\n(Green+1 only)', axes[1])
    plt.colorbar(im2, ax=axes[1], label='Q-value')

    plt.suptitle('Q-value maps: A-1 vs A-2 (v3 agents)', fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(SAVE_DIR, 'q_map_a1_vs_a2.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')

    # Save individual maps too
    for q_grid, name in [(q_a1, 'a1'), (q_a2, 'a2')]:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = plot_q_map(q_grid, f'A-{name[-1]} Q-value map', ax)
        plt.colorbar(im, ax=ax, label='Q-value')
        plt.tight_layout()
        p = os.path.join(SAVE_DIR, f'q_map_{name}.png')
        plt.savefig(p, dpi=150)
        plt.close()
        print(f'Saved: {p}')


if __name__ == '__main__':
    main()
