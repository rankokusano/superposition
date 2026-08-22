"""
v5 §8.4 R3-stay data collection: A-2 static (per-episode uniform-random
position, no movement), matching the ORIGINAL PAPER's exp1_l1 base-
learning condition, but with A-2's vision also saved (self_vision +
other_vision) so it can be used identically to r2_a1random_a2rl for the
Q-value pipeline.

Design rationale (v4_experiment_log.md Sec.8.4): R3-A trained with A-2
already MOVING but process-2's input pinned at a zero vector -- an
architectural mismatch the original paper never had (exp1_l1's A-2 was
genuinely static while its process-2 input was also a constant). This
collects the missing "Q-value version of exp1_l1": A-2 truly static,
so a future R3-stay model's process-2-is-zero design matches its data.

A-2 = StayAgent (config/agent/stay.yml). Its own reset() already calls
FieldObject.random_position() -- i.e. A-2's position is resampled
uniformly across the whole arena at the start of EVERY episode, then
stays fixed for the full episode (StayAgent.step() always returns
[0,0] and never touches self.p). No actor/critic needed for A-2 at all
-- this is architecturally simpler than collect_data_r2.py.

A-1 = same env-driven RandomAgent as all other v4 collections
(self_random_other_stay.yml's agent.self=random).

Sec.5.1-style requirements (same as collect_data_r2.py):
    - saves self_vision AND other_vision (A-2's own camera)
    - does NOT save Q-values -- recomputed later from a critic checkpoint
    - deterministic seeding (seed = md5(seed_key)), recorded as HDF5 attrs

Usage (inside Docker, from /work/my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data_r3_stay.py --dataset r3_stay
"""

import argparse
import hashlib
import os
import random
import sys

import h5py
import numpy as np

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

# matches r2_a1random_a2rl's scale, since R4-move (which continues from
# R3-stay) trains on r2_a1random_a2rl at this scale -- keeps the pretrain
# chain (exp1_l1_1000 -> R3-stay -> R4-move) on a consistent data volume
# rather than introducing a third scale.
DATASET_CONFIGS = {
    'r3_stay': {
        'seq_length': 101,
        'n_data': {'train': 3000, 'test': 300},
    },
}


def get_seed(seed_key):
    return int(hashlib.md5(seed_key.encode('utf-8')).hexdigest()[:8], 16)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def vision_to_uint8(v):
    """v: float32 in [0,1], already 8-bit-quantized at the source."""
    return np.round(v * 255).astype(np.uint8)


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


def collect(env, cfg, h5_file):
    world = env.world
    seq_length = cfg['seq_length']
    cam_h = world.config.camera.agent.height
    cam_w = world.config.camera.agent.width

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
            env.reset()  # A-2 (StayAgent) resamples a uniform-random position here

            for t in range(seq_length):
                sp = env.self_agent.p.copy()
                op = env.other_agent.p.copy()

                # ── A-1 vision (self camera) + A-1 motion (env-driven RandomAgent) ──
                env.world.set_camera('self')
                env.world.draw(env.self_agent, env.other_agent)
                sv = env.world.capture()
                sm = env.self_agent.step()  # advances env.self_agent.p internally

                # ── A-2 vision (other camera); A-2 never moves ──
                env.world.set_camera('other')
                env.world.draw(env.self_agent, env.other_agent)
                ov = env.world.capture()
                om = env.other_agent.step()  # StayAgent.step() == [0,0], p unchanged

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
        h5_file.attrs['a2_behavior'] = 'StayAgent (static, per-episode uniform-random position)'
        collect(env, cfg, h5_file)

    print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()
