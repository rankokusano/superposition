"""
v4 R2 data collection: exp3's exact architecture/data-scale, with A-2's
behaviour swapped from the original paper's periodic/scripted policy to
the trained RL Green-preference policy. This is the ONLY change from
exp3 -- A-1 still moves via the same env-driven RandomAgent used by the
original self_random_other_* collection configs (uniform-random target,
walk-to-target, sleep_p), not a hand-rolled random action.

Sec.5.1 requirements:
    - saves self_vision AND other_vision (A-2's own camera), so a future
      stage can compute A-2's true Q-value from A-2's own critic without
      recollecting
    - does NOT save Q-values -- those are cheap to recompute later from a
      critic checkpoint + vision, and baking them in would lock in
      whatever normalization/probe-count choice is made now
    - action generation for A-2 is the actor's own stochastic sample()
      (not deterministic tanh(mean) -- see 2026-08-12 conversation:
      determinism collapsed A-2's position coverage)
    - deterministic seeding (Sec.5.0.1 scheme): seed = md5(seed_key),
      seed_key defaults to --dataset, recorded as HDF5 attrs

Vision is stored as uint8 (0-255), not float32. env.world.capture() returns
float32 in [0,1] that is already 8-bit-quantized at the source (verified:
values are exact multiples of 1/255), so round(v*255).astype(uint8) loses
nothing. exp/loader.py converts back to [0,1] float before the existing
scale_vision() (v*2-1) call. This is a 4x size reduction (float32 -> uint8)
independent of the disk-space investigation below.

A disk-space precheck aborts before writing if free space is less than
1.5x the dataset's calculated byte size, so a shortfall fails fast with a
clear message instead of a mid-write OSError/HDF5 segfault.

exp3's actual training data ('self_random_other_stay_periodic') was
combine_dataset.py of 3 sub-collections (stay/clockwise/counter_clockwise)
totalling train=3000, test=300, seq_length=101. This script matches that
total episode count in a single collection (A-2's RL policy takes the
place of all three).

Usage (inside Docker, from /work/my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data_r2.py --dataset r2_a1random_a2rl
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
from my_research.rl_agent_sac import ActorLSTM  # noqa

WORK_ROOT = '/work'
DATA_ROOT = os.path.join(WORK_ROOT, 'my_research', 'preference_inference', 'data')

A2_ACTOR_PATH = os.path.join(DATA_ROOT, 'model', 'v3_rl_a2_actor.pth')
ENV_CONFIG = '/work/simulation/config/collect/self_random_other_stay.yml'

VISION_CHANNELS = 3
MOTION_DIM = 2

DATASET_CONFIGS = {
    'r2_a1random_a2rl': {
        'seq_length': 101,
        'n_data': {'train': 3000, 'test': 300},
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


def load_a2_actor(device):
    actor = ActorLSTM().to(device)
    actor.load_state_dict(torch.load(A2_ACTOR_PATH, map_location=device))
    actor.eval()
    return actor


def estimate_bytes(cfg, cam_h, cam_w):
    total = 0
    for n_data in cfg['n_data'].values():
        vision_bytes = n_data * cfg['seq_length'] * cam_h * cam_w * VISION_CHANNELS * 1  # uint8
        motion_bytes = n_data * cfg['seq_length'] * MOTION_DIM * 4  # float32
        total += 2 * vision_bytes + 4 * motion_bytes  # self+other vision, 4 motion/position fields
    return total


def check_disk_space(save_path, required_bytes, safety_factor=1.5):
    stat = os.statvfs(os.path.dirname(save_path))
    free_bytes = stat.f_bavail * stat.f_frsize
    needed = int(required_bytes * safety_factor)
    print(f'Estimated dataset size: {required_bytes / 1e9:.2f} GB  '
          f'(x{safety_factor} safety margin: {needed / 1e9:.2f} GB)')
    print(f'Free disk space: {free_bytes / 1e9:.2f} GB')
    if free_bytes < needed:
        raise RuntimeError(
            f'Not enough disk space: need {needed / 1e9:.2f} GB '
            f'(estimate x{safety_factor}), only {free_bytes / 1e9:.2f} GB free at '
            f'{os.path.dirname(save_path)}. Aborting before writing anything.'
        )


def collect(env, a2_actor, cfg, h5_file, device):
    world = env.world
    seq_length = cfg['seq_length']
    cam_h = world.config.camera.agent.height
    cam_w = world.config.camera.agent.width
    bound = world.get_boundary()
    clip_lo = np.array([bound[0][0], bound[1][0]])
    clip_hi = np.array([bound[0][1], bound[1][1]])

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

        for n in range(n_data):
            env.reset()
            a2_hidden = None

            for t in range(seq_length):
                sp = env.self_agent.p.copy()
                op = env.other_agent.p.copy()

                # ── A-1 vision (self camera) + A-1 motion (env-driven RandomAgent) ──
                env.world.set_camera('self')
                env.world.draw(env.self_agent, env.other_agent)
                sv = env.world.capture()
                sm = env.self_agent.step()  # advances env.self_agent.p internally

                # ── A-2 vision (other camera) + A-2 motion (trained SAC, stochastic) ──
                env.world.set_camera('other')
                env.world.draw(env.self_agent, env.other_agent)
                ov = env.world.capture()
                ov_t = vision_to_tensor(ov, device)
                with torch.no_grad():
                    a2_action_t, _, a2_hidden = a2_actor.sample(ov_t, a2_hidden)
                om = a2_action_t.squeeze(0).cpu().numpy()
                env.other_agent.p = np.clip(op + om, clip_lo, clip_hi)

                # ── Save ──
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
    args = parser.parse_args()

    seed_key = args.seed_key if args.seed_key is not None else args.dataset
    seed = get_seed(seed_key)
    seed_all(seed)

    cfg = DATASET_CONFIGS[args.dataset]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Dataset: {args.dataset}')
    print(f'Seed key: {seed_key!r} -> seed: {seed}')

    env_cfg = load_config(ENV_CONFIG)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()

    a2_actor = load_a2_actor(device)

    save_dir = os.path.join(DATA_ROOT, 'data', args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'data.h5')

    cam_h = env.world.config.camera.agent.height
    cam_w = env.world.config.camera.agent.width
    required_bytes = estimate_bytes(cfg, cam_h, cam_w)
    check_disk_space(save_path, required_bytes)

    with h5py.File(save_path, 'w') as h5_file:
        h5_file.attrs['seed_key'] = seed_key
        h5_file.attrs['seed'] = seed
        h5_file.attrs['dataset'] = args.dataset
        h5_file.attrs['a2_preference'] = 'Green'
        h5_file.attrs['a2_actor_path'] = A2_ACTOR_PATH
        collect(env, a2_actor, cfg, h5_file, device)

    print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()
