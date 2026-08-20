"""
A-2 TARGET_ENTROPY sweep evaluation (2026-08-20).

For a given (actor, critic) pair trained by train_rl_v3_a2.py under some
target_entropy, rolls A-2 out (self stays, A-2 moves via its own trained
policy, same protocol as probe_a1a2_stochastic_rollout.py) and, using the
SAME rendered frames (no extra rendering pass), computes the K=8 probe-Q
vector via THIS critic at every visited state. Reports:
  - raw (pre-tanh) Q std/range over the actually-visited states
  - Q normalized via A-1's SHARED (mu, sigma) -- unchanged, per Sec.4.2 --
    std/range (the quantity that has to reach >=0.2, ideally >=0.3)
  - position std + per-landmark distance stats (Green-preference check)
  - motion std_x/std_y

This intentionally evaluates each candidate critic against states ITS OWN
paired actor visits (not the old r2 dataset's fixed trajectory), since the
whole point of the sweep is that a higher-entropy policy visits different
states than before -- using the old, narrow-coverage r2 data would just
reproduce the original problem regardless of which critic is asked.

Usage (inside Docker, from /work):
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run --auto-servernum \
        --server-args='-screen 0 1024x768x24' python -u \
        my_research/preference_inference/analyze/a2_critic_sweep_eval.py \
        --actor_path .../v3_rl_a2_actor_tem1p5_seed0.pth \
        --critic_path .../v3_rl_a2_critic_tem1p5_seed0.pth --label te-1.5
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import creator  # noqa
from util import load_config  # noqa
from my_research.rl_agent_sac import ActorLSTM, CriticLSTM  # noqa

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_EPISODES = 50
SEQ_LEN = 101
EPSILON = 0.1  # matches probe_a1a2_stochastic_rollout.py's evaluation convention

A1_MU, A1_SIGMA = 1.9233185578310863, 0.9465510185824275  # A-1's own stats, unchanged (Sec.4.2)

LANDMARKS = {
    'Red':    np.array([-9.0,  9.0]),
    'Green':  np.array([-9.0, -9.0]),
    'Blue':   np.array([ 9.0, -9.0]),
    'Yellow': np.array([ 9.0,  9.0]),
}

SAVE_DIR = 'my_research/preference_inference/data/result/baseline_v4'

K = 8
ANGLES = [2 * math.pi * i / K for i in range(K)]
PROBES = torch.tensor([[math.cos(a), math.sin(a)] for a in ANGLES],
                       dtype=torch.float32, device=DEVICE)


def vision_to_tensor(v):
    v = v.astype(np.float32)
    return torch.tensor(np.transpose(v, (2, 0, 1))).unsqueeze(0).to(DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_path', required=True)
    parser.add_argument('--critic_path', required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()

    env_cfg = load_config('/work/simulation/config/collect/self_stay_other_random.yml')
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()

    actor = ActorLSTM().to(DEVICE)
    actor.load_state_dict(torch.load(args.actor_path, map_location=DEVICE))
    actor.eval()
    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(args.critic_path, map_location=DEVICE))
    critic.eval()

    np.random.seed(12345)
    import random
    random.seed(12345)
    torch.manual_seed(12345)

    all_pos = []
    all_raw_q = []
    with torch.no_grad():
        for ep in range(N_EPISODES):
            env.reset()
            hidden = None
            for t in range(SEQ_LEN):
                env.world.set_camera('other')
                env.world.draw(env.self_agent, env.other_agent)
                v = env.world.capture()
                v_t = vision_to_tensor(v)

                action_t, _, hidden = actor.sample(v_t, hidden)
                action_np = action_t.squeeze(0).cpu().numpy()
                if np.random.rand() < EPSILON:
                    action_np = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
                env.other_agent.p = np.clip(env.other_agent.p + action_np, -9.5, 9.5)
                all_pos.append(env.other_agent.p.copy())

                v_rep = v_t.expand(K, -1, -1, -1)
                q, _ = critic(v_rep, PROBES, hidden=None)
                all_raw_q.append(q.squeeze(-1).cpu().numpy())

    pos = np.array(all_pos).reshape(N_EPISODES, SEQ_LEN, 2)
    raw_q = np.array(all_raw_q)  # (N_EPISODES*SEQ_LEN, K)

    motion = np.diff(pos, axis=1).reshape(-1, 2)
    pos_flat = pos.reshape(-1, 2)

    landmark_stats = {}
    for name, lp in LANDMARKS.items():
        d = np.linalg.norm(pos_flat - lp, axis=1)
        landmark_stats[name] = {
            'mean_dist': float(d.mean()), 'std_dist': float(d.std()),
            'frac_within_1.5': float((d < 1.5).mean()),
            'frac_within_3.0': float((d < 3.0).mean()),
        }

    norm_q = np.tanh((raw_q - A1_MU) / A1_SIGMA)

    result = {
        'label': args.label,
        'actor_path': args.actor_path,
        'critic_path': args.critic_path,
        'n_episodes': N_EPISODES,
        'raw_q_mean': float(raw_q.mean()), 'raw_q_std': float(raw_q.std()),
        'raw_q_min': float(raw_q.min()), 'raw_q_max': float(raw_q.max()),
        'norm_q_mean': float(norm_q.mean()), 'norm_q_std': float(norm_q.std()),
        'norm_q_min': float(norm_q.min()), 'norm_q_max': float(norm_q.max()),
        'position_std': [float(pos_flat[:, 0].std()), float(pos_flat[:, 1].std())],
        'motion_std': [float(motion[:, 0].std()), float(motion[:, 1].std())],
        'motion_mean': [float(motion[:, 0].mean()), float(motion[:, 1].mean())],
        'landmark_stats': landmark_stats,
    }

    print(f"=== {args.label} ===")
    print(f"raw Q:  mean={result['raw_q_mean']:.4f} std={result['raw_q_std']:.4f} "
          f"range=[{result['raw_q_min']:.4f},{result['raw_q_max']:.4f}]")
    print(f"norm Q (A-1 shared stats): mean={result['norm_q_mean']:.4f} "
          f"std={result['norm_q_std']:.4f} range=[{result['norm_q_min']:.4f},{result['norm_q_max']:.4f}]")
    print(f"position std: x={result['position_std'][0]:.3f} y={result['position_std'][1]:.3f}")
    print(f"motion std:   x={result['motion_std'][0]:.4f} y={result['motion_std'][1]:.4f}  "
          f"(mean x={result['motion_mean'][0]:.4f} y={result['motion_mean'][1]:.4f})")
    for name, s in landmark_stats.items():
        print(f"  dist to {name:7s}: mean={s['mean_dist']:6.2f} within1.5={s['frac_within_1.5']*100:5.1f}%")

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'a2_sweep_{args.label}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
