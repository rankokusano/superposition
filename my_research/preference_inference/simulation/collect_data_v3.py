"""
v3 data collection for preference inference.

Datasets:
    v3_b_base_train  : A-1 RANDOM movement (Q¹ from RL critic), A-2 stationary
    v3_b_mgve_train  : A-1 RL policy, A-2 RL policy (green, speed x0.15)
    v3_eval          : A-1 RL policy, A-2 RL policy (green, speed x0.15)

All datasets use seq_length=101 and 2100 training episodes (matching original paper).

Usage (inside Docker):
    cd /work/my_research/preference_inference
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data_v3.py --dataset v3_b_base_train
"""

import argparse
import hashlib
import os
import random
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import creator  # noqa
from util import load_config  # noqa
from my_research.rl_agent_sac import ActorLSTM, CriticLSTM  # noqa

WORK_ROOT = '/work'
DATA_ROOT = os.path.join(WORK_ROOT, 'my_research', 'preference_inference', 'data')

A1_ACTOR_PATH  = os.path.join(DATA_ROOT, 'model', 'v3_rl_actor.pth')
A1_CRITIC_PATH = os.path.join(DATA_ROOT, 'model', 'v3_rl_critic.pth')
A2_ACTOR_PATH  = os.path.join(DATA_ROOT, 'model', 'v3_rl_a2_actor.pth')

ENV_CONFIG_STAY   = '/work/simulation/config/collect/self_random_other_stay.yml'
ENV_CONFIG_RANDOM = '/work/simulation/config/collect/self_stay_other_random.yml'

A2_SPEED = 0.15  # scale A-2's RL action to reduce standing-still

DATASET_CONFIGS = {
    'v3_b_base_train': {
        'a1_mode': 'random',   # A-1 moves randomly (Q¹ still from critic)
        'a2_mode': 'stay',     # A-2 stationary
        'seq_length': 101,
        'n_data': {'train': 2100, 'test': 100},
        'env_config': ENV_CONFIG_STAY,
    },
    'v3_b_mgve_train': {
        'a1_mode': 'rl',       # A-1 follows RL policy
        'a2_mode': 'rl',       # A-2 follows RL policy (green preference, x0.15)
        'seq_length': 101,
        'n_data': {'train': 2100, 'test': 100},
        'env_config': ENV_CONFIG_STAY,
    },
    'v3_eval': {
        'a1_mode': 'rl',
        'a2_mode': 'rl',
        'seq_length': 101,
        'n_data': {'train': 100, 'test': 100},
        'env_config': ENV_CONFIG_STAY,
    },
}

VISION_CHANNELS = 3
MOTION_DIM = 2


def load_a1_agents(device):
    actor = ActorLSTM().to(device)
    actor.load_state_dict(torch.load(A1_ACTOR_PATH, map_location=device))
    actor.eval()
    critic = CriticLSTM().to(device)
    critic.load_state_dict(torch.load(A1_CRITIC_PATH, map_location=device))
    critic.eval()
    return actor, critic


def load_a2_actor(device):
    actor = ActorLSTM().to(device)
    actor.load_state_dict(torch.load(A2_ACTOR_PATH, map_location=device))
    actor.eval()
    return actor


def vision_to_tensor(v, device):
    v = v.astype(np.float32)
    return torch.tensor(np.transpose(v, (2, 0, 1))).unsqueeze(0).to(device)


def get_q1(critic, vision_np, action_np, device):
    v_t = vision_to_tensor(vision_np, device)
    a_t = torch.tensor(action_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q, _ = critic(v_t, a_t, hidden=None)
    return q.squeeze().cpu().item()


def collect(env, a1_actor, a1_critic, a2_actor, cfg, h5_file, device):
    world = env.world
    seq_length = cfg['seq_length']
    cam_h = world.config.camera.agent.height
    cam_w = world.config.camera.agent.width
    bound = world.get_boundary()
    clip_lo = np.array([bound[0][0], bound[1][0]])
    clip_hi = np.array([bound[0][1], bound[1][1]])

    a1_mode = cfg['a1_mode']
    a2_mode = cfg['a2_mode']

    for mode, n_data in cfg['n_data'].items():
        print(f"Collecting [{mode}]: {n_data} episodes x {seq_length} steps")

        ds = {
            'self_vision':    h5_file.create_dataset(
                f'{mode}/self_vision',    (n_data, seq_length, cam_h, cam_w, VISION_CHANNELS), dtype='f'),
            'self_motion':    h5_file.create_dataset(
                f'{mode}/self_motion',    (n_data, seq_length, MOTION_DIM), dtype='f'),
            'self_position':  h5_file.create_dataset(
                f'{mode}/self_position',  (n_data, seq_length, MOTION_DIM), dtype='f'),
            'other_motion':   h5_file.create_dataset(
                f'{mode}/other_motion',   (n_data, seq_length, MOTION_DIM), dtype='f'),
            'other_position': h5_file.create_dataset(
                f'{mode}/other_position', (n_data, seq_length, MOTION_DIM), dtype='f'),
            'a1_q_values':    h5_file.create_dataset(
                f'{mode}/a1_q_values',    (n_data, seq_length, 1), dtype='f'),
        }

        for n in range(n_data):
            env.reset()
            a1_hidden = None
            a2_hidden = None

            for t in range(seq_length):
                # ── A-1 vision (self camera) ──────────────────────────────
                env.world.set_camera('self')
                env.world.draw(env.self_agent, env.other_agent)
                sv = env.world.capture()

                sp = env.self_agent.p.copy()
                op = env.other_agent.p.copy()

                # ── A-1 action ────────────────────────────────────────────
                sv_t = vision_to_tensor(sv, device)
                if a1_mode == 'rl':
                    with torch.no_grad():
                        a1_action_t, _, a1_hidden = a1_actor.sample(sv_t, a1_hidden)
                    a1_action = a1_action_t.squeeze(0).cpu().numpy()
                else:  # random
                    a1_action = np.random.uniform(-1.0, 1.0, size=MOTION_DIM).astype(np.float32)

                # Q¹ always from RL critic (regardless of A-1's movement mode)
                with torch.no_grad():
                    a1_action_t_for_q = torch.tensor(
                        a1_action, dtype=torch.float32).unsqueeze(0).to(device)
                    q1, _ = a1_critic(sv_t, a1_action_t_for_q, hidden=None)
                q_val = q1.squeeze().cpu().item()

                # ── A-2 action ────────────────────────────────────────────
                if a2_mode == 'rl':
                    env.world.set_camera('other')
                    env.world.draw(env.self_agent, env.other_agent)
                    ov = env.world.capture()
                    ov_t = vision_to_tensor(ov, device)
                    with torch.no_grad():
                        a2_action_t, _, a2_hidden = a2_actor.sample(ov_t, a2_hidden)
                    a2_action = a2_action_t.squeeze(0).cpu().numpy() * A2_SPEED
                else:  # stay
                    a2_action = np.zeros(MOTION_DIM, dtype=np.float32)

                # ── Apply actions ─────────────────────────────────────────
                env.self_agent.p  = np.clip(sp + a1_action,  clip_lo, clip_hi)
                env.other_agent.p = np.clip(op + a2_action, clip_lo, clip_hi)

                # ── Save ──────────────────────────────────────────────────
                ds['self_vision'][n, t]    = sv
                ds['self_motion'][n, t]    = a1_action
                ds['self_position'][n, t]  = sp
                ds['other_motion'][n, t]   = a2_action
                ds['other_position'][n, t] = op
                ds['a1_q_values'][n, t]    = np.array([q_val], dtype=np.float32)

            if (n + 1) % 100 == 0:
                print(f"  {mode}: {n+1}/{n_data} episodes done")

        print(f"  [{mode}] done.")


def get_seed(seed_key):
    """Deterministic seed from a string key, same scheme as root repo's
    simulation/collect_data.py (md5(key) -> first 8 hex chars -> int)."""
    return int(hashlib.md5(seed_key.encode('utf-8')).hexdigest()[:8], 16)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--seed_key', default=None,
                         help='String hashed into the seed. Defaults to --dataset, '
                              'so re-running with the same --dataset reproduces the '
                              'same trajectories. Override to collect a distinct seed '
                              'variant of the same dataset (e.g. for repeated-measurement '
                              'baselines).')
    args = parser.parse_args()

    seed_key = args.seed_key if args.seed_key is not None else args.dataset
    seed = get_seed(seed_key)
    seed_all(seed)

    cfg = DATASET_CONFIGS[args.dataset]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Dataset: {args.dataset}')
    print(f'Seed key: {seed_key!r} -> seed: {seed}')

    env_cfg = load_config(cfg['env_config'])
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()

    a1_actor, a1_critic = load_a1_agents(device)
    a2_actor = load_a2_actor(device) if cfg['a2_mode'] == 'rl' else None

    save_dir = os.path.join(DATA_ROOT, 'data', args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'data.h5')

    with h5py.File(save_path, 'w') as h5_file:
        h5_file.attrs['seed_key'] = seed_key
        h5_file.attrs['seed'] = seed
        h5_file.attrs['dataset'] = args.dataset
        collect(env, a1_actor, a1_critic, a2_actor, cfg, h5_file, device)

    print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()
