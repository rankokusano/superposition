"""
v4 Sec.5.2 (revised 2026-08-12) verification, stochastic version.

Determinism was withdrawn: probing showed deterministic tanh(mean) rollout
collapses A-2's position spread relative to the original stochastic
collection, so evaluation/collection should use the actor's own stochastic
sample() (optionally mixed with uniform-random exploration via epsilon).

For both A-1 and A-2, rolls out N_EPISODES x SEQ_LEN steps using:
    with prob EPSILON: uniform-random action in [-1,1]^2
    else:              actor.sample(state, hidden)  (SAC's trained policy)
and reports position mean/std, per-landmark distance stats (all 4
landmarks), and a 2D visitation heatmap.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/probe_a1a2_stochastic_rollout.py [--epsilon 0.1]
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
from my_research.rl_agent_sac import ActorLSTM  # noqa

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_EPISODES = 50
SEQ_LEN = 101

LANDMARKS = {
    'Red':    np.array([-9.0,  9.0]),
    'Green':  np.array([-9.0, -9.0]),
    'Blue':   np.array([ 9.0, -9.0]),
    'Yellow': np.array([ 9.0,  9.0]),  # == "Cyan" in the instruction doc's naming
}

SAVE_DIR = 'data/result/baseline_v4'


def vision_to_tensor(v):
    v = v.astype(np.float32)
    return torch.tensor(np.transpose(v, (2, 0, 1))).unsqueeze(0).to(DEVICE)


def rollout(actor_path, env_config, move_self, camera_key, seed, epsilon):
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.manual_seed(seed)

    env_cfg = load_config(env_config)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()

    actor = ActorLSTM().to(DEVICE)
    actor.load_state_dict(torch.load(actor_path, map_location=DEVICE))
    actor.eval()

    all_pos = []
    for ep in range(N_EPISODES):
        env.reset()
        hidden = None
        for t in range(SEQ_LEN):
            env.world.set_camera(camera_key)
            env.world.draw(env.self_agent, env.other_agent)
            v = env.world.capture()
            v_t = vision_to_tensor(v)
            with torch.no_grad():
                action_t, _, hidden = actor.sample(v_t, hidden)
            action_np = action_t.squeeze(0).cpu().numpy()
            if np.random.rand() < epsilon:
                action_np = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
            agent = env.self_agent if move_self else env.other_agent
            agent.p = np.clip(agent.p + action_np, -9.5, 9.5)
            all_pos.append(agent.p.copy())

    return np.array(all_pos)


def landmark_stats(pos):
    stats = {}
    for name, lp in LANDMARKS.items():
        d = np.linalg.norm(pos - lp, axis=1)
        stats[name] = {
            'mean_dist': float(d.mean()),
            'std_dist': float(d.std()),
            'frac_within_1.5': float((d < 1.5).mean()),
            'frac_within_3.0': float((d < 3.0).mean()),
        }
    return stats


def print_stats(label, pos, stats):
    print(f'\n=== {label} ===')
    print(f'n_samples={len(pos)}  mean=({pos[:,0].mean():.3f}, {pos[:,1].mean():.3f})  '
          f'std=({pos[:,0].std():.3f}, {pos[:,1].std():.3f})')
    for name, s in stats.items():
        print(f'  dist to {name:7s}: mean={s["mean_dist"]:6.2f}  std={s["std_dist"]:5.2f}  '
              f'within1.5={s["frac_within_1.5"]*100:5.1f}%  within3.0={s["frac_within_3.0"]*100:5.1f}%')


def heatmap(pos, label, save_path, bins=20):
    fig, ax = plt.subplots(figsize=(5, 5))
    h = ax.hist2d(pos[:, 0], pos[:, 1], bins=bins, range=[[-9.5, 9.5], [-9.5, 9.5]], cmap='viridis')
    plt.colorbar(h[3], ax=ax, label='visit count')
    for name, lp in LANDMARKS.items():
        ax.plot(lp[0], lp[1], 'w+', markersize=14, markeredgewidth=2)
        ax.annotate(name, lp, color='white', fontsize=8, ha='center', va='bottom')
    ax.set_title(label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f'Saved heatmap: {save_path}')

    counts, xedges, yedges = np.histogram2d(
        pos[:, 0], pos[:, 1], bins=10, range=[[-9.5, 9.5], [-9.5, 9.5]])
    print(f'  text grid (10x10, rows=y high->low, cols=x low->high), counts:')
    for row in counts.T[::-1]:
        print('   ', ' '.join(f'{int(c):4d}' for c in row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=0.1,
                         help='probability of a uniform-random action instead of actor.sample()')
    parser.add_argument('--a1_path', default='/work/my_research/preference_inference/data/model/v3_rl_actor.pth')
    parser.add_argument('--a2_path', default='/work/my_research/preference_inference/data/model/v3_rl_a2_actor.pth')
    parser.add_argument('--label', default='',
                         help='suffix for saved heatmap filenames, e.g. "_seed0"')
    parser.add_argument('--only', choices=['a1', 'a2', 'both'], default='both')
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f'epsilon={args.epsilon}')

    if args.only in ('a1', 'both'):
        pos_a1 = rollout(args.a1_path, '/work/simulation/config/collect/self_random_other_stay.yml',
                          move_self=True, camera_key='self', seed=12345, epsilon=args.epsilon)
        stats_a1 = landmark_stats(pos_a1)
        print_stats(f'A-1{args.label} (Red-seeking, continuous reward, stochastic)', pos_a1, stats_a1)
        heatmap(pos_a1, f'A-1{args.label} position visits (continuous reward, stochastic)',
                os.path.join(SAVE_DIR, f'a1{args.label}_heatmap_stochastic.png'))

    if args.only in ('a2', 'both'):
        pos_a2 = rollout(args.a2_path, '/work/simulation/config/collect/self_stay_other_random.yml',
                          move_self=False, camera_key='other', seed=12345, epsilon=args.epsilon)
        stats_a2 = landmark_stats(pos_a2)
        print_stats(f'A-2{args.label} (Green-seeking, continuous reward, stochastic)', pos_a2, stats_a2)
        heatmap(pos_a2, f'A-2{args.label} position visits (continuous reward, stochastic)',
                os.path.join(SAVE_DIR, f'a2{args.label}_heatmap_stochastic.png'))

    if args.only == 'both':
        print('\n=== side by side ===')
        print(f'A-1 std=({pos_a1[:,0].std():.2f},{pos_a1[:,1].std():.2f})  -> Red   within1.5={stats_a1["Red"]["frac_within_1.5"]*100:5.1f}%')
        print(f'A-2 std=({pos_a2[:,0].std():.2f},{pos_a2[:,1].std():.2f})  -> Green within1.5={stats_a2["Green"]["frac_within_1.5"]*100:5.1f}%')


if __name__ == '__main__':
    main()
