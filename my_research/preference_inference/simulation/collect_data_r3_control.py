"""
v4 R3-A control experiment (2026-08-14): identical to collect_data_r2.py
in every respect except A-2's behaviour -- CyclerA2 (deterministic,
position-independent of A-1) instead of the trained RL Green policy.
Tests whether h1->other's elevated leakage (R2: 0.4929, R3-A: 0.5820,
both far above exp1_l1/exp3's ~0.01-0.04) is caused by A-2 being a
value-driven, visually-reactive RL agent, or by the Q-value input itself.
If Cycler restores h1->other to the exp1_l1/exp3 level under the SAME
R3-A architecture, the leakage is behavioral (A-2's RL policy), not
architectural (Q-value input).

A-1: same env-driven RandomAgent as collect_data_r2.py (self_random_
    other_stay.yml, self_agent.step()).
A-2: CyclerA2 (same class/constants as simulation/collect_data.py:
    Red -> Green -> Blue -> Yellow -> Red, GOAL_MARGIN=1.5, SPEED=1.0).

Usage (inside Docker, from /work/my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data_r3_control.py --dataset r3_control_a1random_a2cycler
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

WORK_ROOT = '/work'
DATA_ROOT = os.path.join(WORK_ROOT, 'my_research', 'preference_inference', 'data')

ENV_CONFIG = '/work/simulation/config/collect/self_random_other_stay.yml'

VISION_CHANNELS = 3
MOTION_DIM = 2

DATASET_CONFIGS = {
    'r3_control_a1random_a2cycler': {
        'seq_length': 101,
        'n_data': {'train': 3000, 'test': 300},
    },
}

# -- CyclerA2, same constants/logic as simulation/collect_data.py --
LANDMARK_POSITIONS = {
    'Red':    np.array([-9.0,  9.0]),
    'Green':  np.array([-9.0, -9.0]),
    'Blue':   np.array([ 9.0, -9.0]),
    'Yellow': np.array([ 9.0,  9.0]),
}
LANDMARK_NAMES = ['Red', 'Green', 'Blue', 'Yellow']
LANDMARK_ID = {name: i for i, name in enumerate(LANDMARK_NAMES)}
GOAL_MARGIN = 1.5
CYCLER_SPEED = 1.0


class CyclerA2:
    """A-2 cycles through all 4 landmarks in fixed order indefinitely."""

    def __init__(self):
        self._order = LANDMARK_NAMES[:]
        self._idx = 0
        self._target_name = None
        self._target_pos = None

    def reset(self):
        self._idx = 0
        self._target_name = self._order[0]
        self._target_pos = LANDMARK_POSITIONS[self._target_name].copy()

    def get_action(self, current_pos):
        direction = self._target_pos - current_pos
        distance = np.linalg.norm(direction)
        if distance < GOAL_MARGIN:
            self._idx = (self._idx + 1) % len(self._order)
            self._target_name = self._order[self._idx]
            self._target_pos = LANDMARK_POSITIONS[self._target_name].copy()
            direction = self._target_pos - current_pos
            distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        return (direction / distance * CYCLER_SPEED).astype(np.float32)


def get_seed(seed_key):
    return int(hashlib.md5(seed_key.encode('utf-8')).hexdigest()[:8], 16)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def vision_to_uint8(v):
    return np.round(v * 255).astype(np.uint8)


def estimate_bytes(cfg, cam_h, cam_w):
    total = 0
    for n_data in cfg['n_data'].values():
        vision_bytes = n_data * cfg['seq_length'] * cam_h * cam_w * VISION_CHANNELS * 1
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
        raise RuntimeError(
            f'Not enough disk space: need {needed / 1e9:.2f} GB, only '
            f'{free_bytes / 1e9:.2f} GB free at {os.path.dirname(save_path)}.'
        )


def collect(env, cfg, h5_file):
    world = env.world
    seq_length = cfg['seq_length']
    cam_h = world.config.camera.agent.height
    cam_w = world.config.camera.agent.width
    bound = world.get_boundary()
    clip_lo = np.array([bound[0][0], bound[1][0]])
    clip_hi = np.array([bound[0][1], bound[1][1]])

    cycler = CyclerA2()

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
            cycler.reset()

            for t in range(seq_length):
                sp = env.self_agent.p.copy()
                op = env.other_agent.p.copy()

                env.world.set_camera('self')
                env.world.draw(env.self_agent, env.other_agent)
                sv = env.world.capture()
                sm = env.self_agent.step()  # env-driven RandomAgent

                env.world.set_camera('other')
                env.world.draw(env.self_agent, env.other_agent)
                ov = env.world.capture()
                om = cycler.get_action(op)
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
    args = parser.parse_args()

    seed_key = args.seed_key if args.seed_key is not None else args.dataset
    seed = get_seed(seed_key)
    seed_all(seed)

    cfg = DATASET_CONFIGS[args.dataset]
    print(f'Dataset: {args.dataset}')
    print(f'Seed key: {seed_key!r} -> seed: {seed}')

    env_cfg = load_config(ENV_CONFIG)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()

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
        h5_file.attrs['a2_policy'] = 'CyclerA2 (Red->Green->Blue->Yellow)'
        collect(env, cfg, h5_file)

    print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()
