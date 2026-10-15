"""
v10 data collection (docs/v10_instructions.md Sec.6): a copy of
collect_data_r2.py that renders with the v10 simulator copy (sim_v10/) and
selects the landmark layout. Everything else -- A-1 RandomAgent, A-2's
stochastic SAC actor (other camera, hidden carried over, position clipped
to the arena), uint8 vision, other_vision saved, no Q-values saved, seed =
md5(seed_key) -- is identical to collect_data_r2.py.

Layout modes (--layout):
    original  every episode uses the original layout
    swap_rg   every episode uses the red/green-swapped layout
    mix       each episode is swap_rg with probability 1/2, drawn from a
              dedicated RandomState(md5(seed_key + '/layout')) so the
              environment's own random stream is untouched
The per-episode layout is saved as `<split>/layout` (0 = original,
1 = swap_rg; sim_v10/layout.py LAYOUT_CODE) in every mode.

--a1_policy chooses how A-1 moves. Only 'random' (the env-driven
RandomAgent, the default and the only setting v10 uses) is implemented; the
argument exists so v11 can add policies without touching v10 behaviour.

Usage (inside Docker, from /work/my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data_v10.py --dataset v10_swap_r2
"""

import argparse
import hashlib
import os
import random
import sys

import h5py
import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/work')
sys.path.insert(0, _PI_DIR)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa
from my_research.rl_agent_sac import ActorLSTM  # noqa  (read-only import)

DATA_ROOT = os.path.join(_PI_DIR, 'data')
ENV_CONFIG = os.path.join(_PI_DIR, 'sim_v10', 'config', 'collect',
                          'self_random_other_stay.yml')
V3_A2_ACTOR = os.path.join(DATA_ROOT, 'model', 'v3_rl_a2_actor.pth')

VISION_CHANNELS = 3
MOTION_DIM = 2

DATASET_CONFIGS = {
    # gate 6.0: first train episodes of r2_a1random_a2rl, same seed key
    'v10_bitcheck_original': {
        'seq_length': 101, 'n_data': {'train': 3},
        'layout': layout.ORIGINAL, 'seed_key': 'r2_a1random_a2rl',
        'a2_actor': V3_A2_ACTOR,
        'save_path': os.path.join(DATA_ROOT, 'result', 'v10_swap', 'gate60',
                                  'bitcheck_original.h5'),
    },
    # stage A, pairing check: swap layout with r2's seed key (3 episodes)
    'v10_pairseed_check_swap': {
        'seq_length': 101, 'n_data': {'train': 3},
        'layout': layout.SWAP_RG, 'seed_key': 'r2_a1random_a2rl',
        'a2_actor': V3_A2_ACTOR,
        'save_path': os.path.join(DATA_ROOT, 'result', 'v10_swap', 'stageA',
                                  'pairseed_check_swap.h5'),
    },
    # stage A
    # Seeded with r2_a1random_a2rl's key (not md5 of its own name; user's
    # decision 2026-10-15, v10_experiment_log Sec.5.11) so A-1's path and both
    # initial positions are identical episode by episode to r2_a1random_a2rl
    # (A0 and A1 pair up per episode). Verified on 3 episodes beforehand.
    'v10_swap_r2': {
        'seq_length': 101, 'n_data': {'train': 3000, 'test': 300},
        'layout': layout.SWAP_RG, 'a2_actor': V3_A2_ACTOR,
        'seed_key': 'r2_a1random_a2rl',
    },
    # stage B2 (A-2 actor fixed after B1, passed with --a2_actor)
    'v10_swapmix_r2': {
        'seq_length': 101, 'n_data': {'train': 3000, 'test': 300},
        'layout': 'mix', 'a2_actor': None,
    },
}


def get_seed(seed_key):
    return int(hashlib.md5(seed_key.encode('utf-8')).hexdigest()[:8], 16)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def vision_to_tensor(v, device):
    v = v.astype(np.float32)
    return torch.tensor(np.transpose(v, (2, 0, 1))).unsqueeze(0).to(device)


def vision_to_uint8(v):
    """v: float32 in [0,1], already 8-bit-quantized at the source."""
    return np.round(v * 255).astype(np.uint8)


def load_a2_actor(path, device):
    actor = ActorLSTM().to(device)
    actor.load_state_dict(torch.load(path, map_location=device))
    actor.eval()
    return actor


def estimate_bytes(cfg, cam_h, cam_w):
    total = 0
    for n_data in cfg['n_data'].values():
        vision_bytes = n_data * cfg['seq_length'] * cam_h * cam_w * VISION_CHANNELS
        motion_bytes = n_data * cfg['seq_length'] * MOTION_DIM * 4
        total += 2 * vision_bytes + 4 * motion_bytes
    return total


def check_disk_space(save_path, required_bytes, safety_factor=1.5):
    stat = os.statvfs(os.path.dirname(save_path))
    free_bytes = stat.f_bavail * stat.f_frsize
    needed = int(required_bytes * safety_factor)
    print(f'Estimated dataset size: {required_bytes / 1e9:.2f} GB  '
          f'(x{safety_factor} safety margin: {needed / 1e9:.2f} GB)')
    print(f'Free disk space: {free_bytes / 1e9:.2f} GB')
    if free_bytes < needed:
        raise RuntimeError('Not enough disk space; aborting before writing.')


def episode_layouts(mode, n_total, layout_seed):
    if mode in layout.LAYOUT_NAMES:
        return [mode] * n_total
    assert mode == 'mix', mode
    rng = np.random.RandomState(layout_seed)
    return [layout.SWAP_RG if rng.rand() < 0.5 else layout.ORIGINAL
            for _ in range(n_total)]


def a1_step(env, a1_policy):
    if a1_policy == 'random':
        return env.self_agent.step()  # advances env.self_agent.p internally
    raise ValueError(a1_policy)


def collect(env, a2_actor, cfg, layouts, a1_policy, h5_file, device):
    world = env.world
    seq_length = cfg['seq_length']
    cam_h = world.config.camera.agent.height
    cam_w = world.config.camera.agent.width
    bound = world.get_boundary()
    clip_lo = np.array([bound[0][0], bound[1][0]])
    clip_hi = np.array([bound[0][1], bound[1][1]])

    ep_global = 0
    for mode, n_data in cfg['n_data'].items():
        print(f'Collecting [{mode}]: {n_data} episodes x {seq_length} steps')

        ds = {
            'self_vision':    h5_file.create_dataset(
                f'{mode}/self_vision',    (n_data, seq_length, cam_h, cam_w, VISION_CHANNELS), dtype='uint8'),
            'other_vision':   h5_file.create_dataset(
                f'{mode}/other_vision',   (n_data, seq_length, cam_h, cam_w, VISION_CHANNELS), dtype='uint8'),
            'self_motion':    h5_file.create_dataset(
                f'{mode}/self_motion',    (n_data, seq_length, MOTION_DIM), dtype='f'),
            'self_position':  h5_file.create_dataset(
                f'{mode}/self_position',  (n_data, seq_length, MOTION_DIM), dtype='f'),
            'other_motion':   h5_file.create_dataset(
                f'{mode}/other_motion',   (n_data, seq_length, MOTION_DIM), dtype='f'),
            'other_position': h5_file.create_dataset(
                f'{mode}/other_position', (n_data, seq_length, MOTION_DIM), dtype='f'),
        }
        ds_layout = h5_file.create_dataset(f'{mode}/layout', (n_data,), dtype='uint8')

        for n in range(n_data):
            ep_layout = layouts[ep_global]
            ep_global += 1
            env.set_layout(ep_layout)
            ds_layout[n] = layout.LAYOUT_CODE[ep_layout]

            env.reset()
            a2_hidden = None

            for t in range(seq_length):
                sp = env.self_agent.p.copy()
                op = env.other_agent.p.copy()

                env.world.set_camera('self')
                env.world.draw(env.self_agent, env.other_agent)
                sv = env.world.capture()
                sm = a1_step(env, a1_policy)

                env.world.set_camera('other')
                env.world.draw(env.self_agent, env.other_agent)
                ov = env.world.capture()
                ov_t = vision_to_tensor(ov, device)
                with torch.no_grad():
                    a2_action_t, _, a2_hidden = a2_actor.sample(ov_t, a2_hidden)
                om = a2_action_t.squeeze(0).cpu().numpy()
                env.other_agent.p = np.clip(op + om, clip_lo, clip_hi)

                ds['self_vision'][n, t] = vision_to_uint8(sv)
                ds['other_vision'][n, t] = vision_to_uint8(ov)
                ds['self_motion'][n, t] = sm
                ds['self_position'][n, t] = sp
                ds['other_motion'][n, t] = om
                ds['other_position'][n, t] = op

            if (n + 1) % 200 == 0:
                print(f'  {mode}: {n + 1}/{n_data} episodes done')

        print(f'  [{mode}] done.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--seed_key', default=None)
    parser.add_argument('--a2_actor', default=None,
                        help='A-2 actor checkpoint (required for v10_swapmix_r2)')
    parser.add_argument('--a1_policy', default='random', choices=['random'],
                        help="how A-1 moves; v10 uses only 'random' (RandomAgent)")
    parser.add_argument('--save_path', default=None)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    seed_key = args.seed_key or cfg.get('seed_key') or args.dataset
    seed = get_seed(seed_key)
    seed_all(seed)
    layout_seed = get_seed(seed_key + '/layout')
    a2_path = args.a2_actor or cfg['a2_actor']
    assert a2_path is not None, 'pass --a2_actor'
    save_path = args.save_path or cfg.get('save_path') or os.path.join(
        DATA_ROOT, 'data', args.dataset, 'data.h5')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  Dataset: {args.dataset}  Seed key: {seed_key!r} -> {seed}')
    print(f'Layout mode: {cfg["layout"]}  A-2 actor: {a2_path}  A-1 policy: {args.a1_policy}')

    env_cfg = load_config(ENV_CONFIG)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()

    a2_actor = load_a2_actor(a2_path, device)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cam_h = env.world.config.camera.agent.height
    cam_w = env.world.config.camera.agent.width
    check_disk_space(save_path, estimate_bytes(cfg, cam_h, cam_w))

    n_total = sum(cfg['n_data'].values())
    layouts = episode_layouts(cfg['layout'], n_total, layout_seed)

    with h5py.File(save_path, 'w') as h5_file:
        h5_file.attrs['seed_key'] = seed_key
        h5_file.attrs['seed'] = seed
        h5_file.attrs['dataset'] = args.dataset
        h5_file.attrs['a2_preference'] = 'Green'
        h5_file.attrs['a2_actor_path'] = a2_path
        h5_file.attrs['a1_policy'] = args.a1_policy
        h5_file.attrs['layout_mode'] = cfg['layout']
        h5_file.attrs['layout_seed'] = layout_seed
        h5_file.attrs['simulator'] = 'sim_v10'
        collect(env, a2_actor, cfg, layouts, args.a1_policy, h5_file, device)

    print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()
