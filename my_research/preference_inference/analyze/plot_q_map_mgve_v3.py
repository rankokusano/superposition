"""
Plot Q̂² heatmap from v3 B-MGVE model in the same format as the v2 Q-map.

For each fixed A-2 position (near Green / near 3 other corners),
sweep A-1 over a 20x20 grid and record Q̂² from the VE module.
This mirrors the v2 format which showed Q¹ vs A-1 position for each A-2 position.

Usage (from /work/my_research/preference_inference inside Docker):
    python analyze/plot_q_map_mgve_v3.py
"""

import os, sys
sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import creator
from util import load_config
from my_research.rl_agent_sac import ActorLSTM, CriticLSTM

# Load preference_inference modules after env imports to avoid simulation conflict
sys.path.insert(0, '/work/my_research/preference_inference')
import model as models

# ── paths ────────────────────────────────────────────────────────────────────
DATA_ROOT   = '/work/my_research/preference_inference/data'
MODEL_PATH  = os.path.join(DATA_ROOT, 'result', 'v3_exp_b_mgve', '0', 'model', '00200.pth')
A1_ACTOR    = os.path.join(DATA_ROOT, 'model', 'v3_rl_actor.pth')
A1_CRITIC   = os.path.join(DATA_ROOT, 'model', 'v3_rl_critic.pth')
ENV_CONFIG  = '/work/simulation/config/collect/self_random_other_stay.yml'
MODEL_CFG   = '/work/my_research/preference_inference/config/model/SuperpositionNetworkApproachBMGVEV3/default.yml'
SAVE_DIR    = os.path.join(DATA_ROOT, 'result', 'q_map_v3_mgve')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRID_N = 20

LANDMARK_POSITIONS = {
    'Green':  np.array([-9.0, -9.0]),   # A-2's preferred landmark
    'Red':    np.array([-9.0,  9.0]),
    'Blue':   np.array([ 9.0, -9.0]),
    'Cyan':   np.array([ 9.0,  9.0]),
}
LANDMARK_COLORS = {'Green': 'green', 'Red': 'red', 'Blue': 'blue', 'Cyan': 'cyan'}


def vision_to_tensor(v):
    return torch.tensor(v.astype(np.float32).transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── load env ──────────────────────────────────────────────────────────────
    cfg = load_config(ENV_CONFIG)
    env = creator.create_environment(cfg.environment)
    env.init()
    env.off_display()

    # ── load v3 B-MGVE model ──────────────────────────────────────────────────
    model_config = load_config(MODEL_CFG)
    mgve_model = models.SuperpositionNetworkApproachBMGVEV3(model_config)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    mgve_model.load_state_dict(ckpt['model'])
    mgve_model.to(DEVICE)
    mgve_model.eval()

    # ── load A-1 RL agents (for Q¹ input) ────────────────────────────────────
    a1_actor  = ActorLSTM().to(DEVICE)
    a1_critic = CriticLSTM().to(DEVICE)
    a1_actor.load_state_dict(torch.load(A1_ACTOR, map_location=DEVICE))
    a1_critic.load_state_dict(torch.load(A1_CRITIC, map_location=DEVICE))
    a1_actor.eval(); a1_critic.eval()

    # ── grid setup ────────────────────────────────────────────────────────────
    bound  = env.world.get_boundary()
    xs = np.linspace(bound[0][0], bound[0][1], GRID_N)
    ys = np.linspace(bound[1][0], bound[1][1], GRID_N)

    # ── compute Q̂² grid for each A-2 position ────────────────────────────────
    q2hat_grids = {}

    for lm_name, a2_pos in LANDMARK_POSITIONS.items():
        print(f'Computing Q̂² grid: A-2 near {lm_name} ...')
        env.other_agent.p = a2_pos.copy()
        q2hat_grid = np.zeros((GRID_N, GRID_N))

        with torch.no_grad():
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    env.self_agent.p = np.array([x, y])
                    env.world.set_camera('self')
                    env.world.draw(env.self_agent, env.other_agent)
                    sv = env.world.capture()

                    sv_t = vision_to_tensor(sv)

                    # Q¹ from RL critic
                    a1_act, _, _ = a1_actor.sample(sv_t, hidden=None)
                    q1, _        = a1_critic(sv_t, a1_act, hidden=None)
                    q1_val = q1.squeeze()  # scalar

                    # Run B-MGVE model (t=0, no masking)
                    mgve_model.init_state(1)
                    x_batch = {
                        'self_vision':  sv_t,
                        'a1_q_values':  q1_val.unsqueeze(0).unsqueeze(0),
                    }
                    pred = mgve_model(x_batch,
                                     p_mask_vision_self=0,
                                     p_mask_vision_other=0)
                    q2hat_grid[i, j] = pred['q2_hat'].squeeze().cpu().item()

        q2hat_grids[lm_name] = q2hat_grid
        print(f'  Q̂² range: [{q2hat_grid.min():.2f}, {q2hat_grid.max():.2f}]')

    # ── plot: same format as v2 ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    xmin, xmax = bound[0]
    ymin, ymax = bound[1]

    for ax, (lm_name, q_grid) in zip(axes, q2hat_grids.items()):
        im = ax.imshow(q_grid.T, origin='lower', cmap='hot',
                       extent=[xmin, xmax, ymin, ymax], aspect='equal')
        ax.set_title(f'A-2 near {lm_name}')
        ax.set_xlabel('A-1 sx')
        ax.set_ylabel('A-1 sy')

        # Mark A-2 actual position
        a2p = LANDMARK_POSITIONS[lm_name]
        ax.scatter([a2p[0]], [a2p[1]], c='cyan', s=120, marker='*', zorder=5,
                   label=f'A-2 ({a2p[0]:.0f},{a2p[1]:.0f})')

        # Mark landmarks
        for ln, lp in LANDMARK_POSITIONS.items():
            ax.scatter(lp[0], lp[1], c=LANDMARK_COLORS[ln], s=60, marker='D',
                       edgecolors='white', linewidths=0.5, zorder=4)
            ax.text(lp[0] + 0.4, lp[1] + 0.4, ln, fontsize=7, color='white')

        plt.colorbar(im, ax=ax, label='Q̂²_t (VE output)')
        ax.legend(fontsize=7, loc='upper right')

    plt.suptitle('v3 B-MGVE: Q̂² heatmap over A-1 positions\n(same format as v2 Q-map)', fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'q2hat_map_by_a2_position.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')

    # ── mean Q̂² vs A-2 position ───────────────────────────────────────────────
    # Re-compute: sweep A-2 over full grid, A-1 fixed at center
    print('\nComputing mean Q̂² vs A-2 position (A-1 at center)...')
    env.self_agent.p = np.array([0.0, 0.0])
    env.world.set_camera('self')
    q2hat_a2grid = np.zeros((GRID_N, GRID_N))

    with torch.no_grad():
        # Precompute Q¹ with A-1 at center (no A-2 position effect on Q¹ here)
        env.other_agent.p = np.array([0.0, 0.0])
        env.world.draw(env.self_agent, env.other_agent)
        sv = env.world.capture()
        sv_t = vision_to_tensor(sv)
        a1_act, _, _ = a1_actor.sample(sv_t, hidden=None)
        q1, _ = a1_critic(sv_t, a1_act, hidden=None)
        q1_val = q1.squeeze()

        for i, ox in enumerate(xs):
            for j, oy in enumerate(ys):
                env.other_agent.p = np.array([ox, oy])
                env.world.set_camera('self')
                env.world.draw(env.self_agent, env.other_agent)
                sv = env.world.capture()
                sv_t = vision_to_tensor(sv)

                mgve_model.init_state(1)
                x_batch = {
                    'self_vision': sv_t,
                    'a1_q_values': q1_val.unsqueeze(0).unsqueeze(0),
                }
                pred = mgve_model(x_batch, p_mask_vision_self=0, p_mask_vision_other=0)
                q2hat_a2grid[i, j] = pred['q2_hat'].squeeze().cpu().item()

    print(f'  Q̂² range: [{q2hat_a2grid.min():.2f}, {q2hat_a2grid.max():.2f}]')

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(q2hat_a2grid.T, origin='lower', cmap='hot',
                   extent=[xmin, xmax, ymin, ymax], aspect='equal')
    plt.colorbar(im, ax=ax, label='Q̂² (VE output)')
    ax.set_title('Mean Q̂² as function of A-2 position\n(A-1 fixed at center)')
    ax.set_xlabel('A-2 position x')
    ax.set_ylabel('A-2 position y')
    for ln, lp in LANDMARK_POSITIONS.items():
        ax.scatter(lp[0], lp[1], c=LANDMARK_COLORS[ln], s=80, marker='D',
                   edgecolors='white', linewidths=0.5, zorder=4)
        ax.text(lp[0] + 0.4, lp[1] + 0.4, ln, fontsize=8, color='white')
    plt.tight_layout()
    p2 = os.path.join(SAVE_DIR, 'q2hat_mean_by_a2_position.png')
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f'Saved: {p2}')


if __name__ == '__main__':
    main()
