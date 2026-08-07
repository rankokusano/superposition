"""
v4 §5.0.1 verification: same seed -> same trajectory.

Runs a tiny (n_data small) collection twice with the identical seed_key and
asserts self_position / other_position / self_motion / other_motion /
a1_q_values are bit-identical. Writes only to /tmp, never touches the real
dataset directories under data/data/.

Usage (inside Docker, from /work/my_research/preference_inference):
    python simulation/verify_seed_determinism.py
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import creator  # noqa
from util import load_config  # noqa

from collect_data_v3 import (  # noqa
    collect, get_seed, load_a1_agents, load_a2_actor, seed_all,
)

VERIFY_CFG = {
    'a1_mode': 'rl',
    'a2_mode': 'rl',
    'seq_length': 20,
    'n_data': {'eval': 3},
    'env_config': '/work/simulation/config/collect/self_random_other_stay.yml',
}

SEED_KEY = 'verify_seed_determinism'
OUT_DIR = '/tmp/verify_seed_determinism'


def run_once(device, out_path):
    seed = get_seed(SEED_KEY)
    seed_all(seed)
    print(f'seed_key={SEED_KEY!r} -> seed={seed}')

    env_cfg = load_config(VERIFY_CFG['env_config'])
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()

    a1_actor, a1_critic = load_a1_agents(device)
    a2_actor = load_a2_actor(device)

    with h5py.File(out_path, 'w') as h5_file:
        collect(env, a1_actor, a1_critic, a2_actor, VERIFY_CFG, h5_file, device)


def main():
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(OUT_DIR, exist_ok=True)
    path_a = os.path.join(OUT_DIR, 'run_a.h5')
    path_b = os.path.join(OUT_DIR, 'run_b.h5')

    print('=== run A ===')
    run_once(device, path_a)
    print('=== run B ===')
    run_once(device, path_b)

    fields = ['self_position', 'other_position', 'self_motion', 'other_motion', 'a1_q_values']
    all_match = True
    with h5py.File(path_a, 'r') as fa, h5py.File(path_b, 'r') as fb:
        for field in fields:
            a = fa[f'eval/{field}'][()]
            b = fb[f'eval/{field}'][()]
            identical = np.array_equal(a, b)
            max_abs_diff = np.max(np.abs(a - b)) if not identical else 0.0
            print(f'{field}: identical={identical} max_abs_diff={max_abs_diff}')
            all_match = all_match and identical

    print()
    print('RESULT:', 'PASS (bit-identical trajectories)' if all_match else 'FAIL (trajectories diverged)')
    sys.exit(0 if all_match else 1)


if __name__ == '__main__':
    main()
