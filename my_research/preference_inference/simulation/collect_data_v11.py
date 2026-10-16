"""
v11 stage-1 data collection (docs/v11_experiment_log.md Sec.4.2, 4.6-4.9,
5.1). A copy of collect_data_v10.py with:

  * no A-2 actor: A-2 is the StayAgent of the paper's
    self_random_other_stay setting (random position per episode, static);
    --a2_policy exists for stage 2 but only 'stay' is implemented/used;
  * A-1 moves with --a1_policy random (RandomAgent, paper) or mixed
    (per awake step: p_rule visual-rule step toward the liked colour, else
    the RandomAgent step; v11core/rule_agent.py);
  * per-episode re-seeding of python `random` and numpy's global RNG from
    md5('<seed_key>/<split>/<n>'), with one seed key shared by both
    conditions (default 'v11_s1'), so episode n starts from the same A-1
    position, first random target and A-2 position in both conditions
    (Sec.4.8; deviates from the "md5 of the dataset name" rule on purpose);
  * layout per episode = mix (1/2) from RandomState(md5(seed_key+'/layout')),
    shared by both conditions;
  * the mixed coin comes from RandomState(md5('<seed_key>/<split>/<n>/a1_mix'));
  * per-step `a1_source` (0 random, 1 rule, 2 rule->random fallback,
    3 asleep) and per-episode `layout` are saved;
  * self_vision and other_vision are both captured at the pre-step
    positions (the paper's collect_data.py order; v10 captured other_vision
    after A-1's move because A-2's actor acted on it).

Vision is uint8; no pleasure values are saved (computed from vision).

Usage (inside Docker, from the v11 worktree's my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data_v11.py --dataset v11_s1_random
"""

import argparse
import hashlib
import os
import random
import sys

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PI_DIR)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa
from v11core.rule_agent import AgentPolicy  # noqa

DATA_ROOT = os.path.join(_PI_DIR, 'data')
ENV_CONFIG = os.path.join(_PI_DIR, 'sim_v10', 'config', 'collect',
                          'self_random_other_stay.yml')

VISION_CHANNELS = 3
MOTION_DIM = 2
SEED_KEY = 'v11_s1'
FULL = {'train': 3000, 'test': 300}
GATE61 = os.path.join(DATA_ROOT, 'result', 'v11', 'gate61')

DATASET_CONFIGS = {
    'v11_s1_random': {'a1_policy': 'random', 'n_data': FULL},
    'v11_s1_mixed': {'a1_policy': 'mixed', 'n_data': FULL},
}
# gate 6.1 item 1: first 3 train episodes, collected twice (a/b)
for _cond in ('random', 'mixed'):
    for _rep in ('a', 'b'):
        DATASET_CONFIGS[f'v11_s1_{_cond}_check_{_rep}'] = {
            'a1_policy': _cond, 'n_data': {'train': 3},
            'save_path': os.path.join(GATE61, f'check_{_cond}_{_rep}.h5'),
        }


def get_seed(key):
    return int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16)


def vision_to_uint8(v):
    """v: float32 in [0,1], already 8-bit-quantized at the source."""
    return np.round(v * 255).astype(np.uint8)


def check_disk_space(save_path, n_data, seq_length, cam_h, cam_w, safety_factor=1.5):
    n = sum(n_data.values())
    required = n * seq_length * (2 * cam_h * cam_w * VISION_CHANNELS + 4 * MOTION_DIM * 4 + 1)
    stat = os.statvfs(os.path.dirname(save_path))
    free = stat.f_bavail * stat.f_frsize
    print(f'Estimated dataset size: {required / 1e9:.2f} GB, free {free / 1e9:.1f} GB')
    if free < required * safety_factor:
        raise RuntimeError('Not enough disk space; aborting before writing.')


def episode_layouts(n_total, seed_key):
    # Same draw order as collect_data_v10.episode_layouts('mix', ...).
    rng = np.random.RandomState(get_seed(seed_key + '/layout'))
    return [layout.SWAP_RG if rng.rand() < 0.5 else layout.ORIGINAL
            for _ in range(n_total)]


def collect(env, policy, n_data_cfg, layouts, seq_length, seed_key, h5_file):
    world = env.world
    cam_h = world.config.camera.agent.height
    cam_w = world.config.camera.agent.width

    ep_global = 0
    for split, n_data in n_data_cfg.items():
        print(f'Collecting [{split}]: {n_data} episodes x {seq_length} steps')
        vshape = (n_data, seq_length, cam_h, cam_w, VISION_CHANNELS)
        mshape = (n_data, seq_length, MOTION_DIM)
        ds = {
            'self_vision': h5_file.create_dataset(f'{split}/self_vision', vshape, dtype='uint8'),
            'other_vision': h5_file.create_dataset(f'{split}/other_vision', vshape, dtype='uint8'),
            'self_motion': h5_file.create_dataset(f'{split}/self_motion', mshape, dtype='f'),
            'self_position': h5_file.create_dataset(f'{split}/self_position', mshape, dtype='f'),
            'other_motion': h5_file.create_dataset(f'{split}/other_motion', mshape, dtype='f'),
            'other_position': h5_file.create_dataset(f'{split}/other_position', mshape, dtype='f'),
            'a1_source': h5_file.create_dataset(f'{split}/a1_source', (n_data, seq_length), dtype='int8'),
        }
        ds_layout = h5_file.create_dataset(f'{split}/layout', (n_data,), dtype='uint8')

        for n in range(n_data):
            ep_layout = layouts[ep_global]
            ep_global += 1
            env.set_layout(ep_layout)
            ds_layout[n] = layout.LAYOUT_CODE[ep_layout]

            ep_seed = get_seed(f'{seed_key}/{split}/{n}')
            random.seed(ep_seed)
            np.random.seed(ep_seed)
            policy.set_rng(np.random.RandomState(get_seed(f'{seed_key}/{split}/{n}/a1_mix')))
            env.reset()

            sv_ep = np.empty(vshape[1:], np.uint8)
            ov_ep = np.empty(vshape[1:], np.uint8)
            sm_ep = np.empty(mshape[1:], np.float32)
            sp_ep = np.empty(mshape[1:], np.float32)
            om_ep = np.empty(mshape[1:], np.float32)
            op_ep = np.empty(mshape[1:], np.float32)
            src_ep = np.empty(seq_length, np.int8)
            for t in range(seq_length):
                sp = env.self_agent.p.copy()
                op = env.other_agent.p.copy()

                world.set_camera('self')
                world.draw(env.self_agent, env.other_agent)
                sv = world.capture()
                world.set_camera('other')
                world.draw(env.self_agent, env.other_agent)
                ov = world.capture()

                sm, src = policy.step(sv)
                om = env.other_agent.step()

                sv_ep[t] = vision_to_uint8(sv)
                ov_ep[t] = vision_to_uint8(ov)
                sm_ep[t] = sm
                sp_ep[t] = sp
                om_ep[t] = om
                op_ep[t] = op
                src_ep[t] = src

            ds['self_vision'][n] = sv_ep
            ds['other_vision'][n] = ov_ep
            ds['self_motion'][n] = sm_ep
            ds['self_position'][n] = sp_ep
            ds['other_motion'][n] = om_ep
            ds['other_position'][n] = op_ep
            ds['a1_source'][n] = src_ep

            if (n + 1) % 200 == 0:
                print(f'  {split}: {n + 1}/{n_data} episodes done', flush=True)
        print(f'  [{split}] done.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--seed_key', default=SEED_KEY)
    parser.add_argument('--p_rule', type=float, default=0.5)
    parser.add_argument('--liked', default='Red')
    parser.add_argument('--a2_policy', default='stay', choices=['stay'],
                        help="how A-2 moves; stage 1 uses only 'stay' (StayAgent)")
    parser.add_argument('--save_path', default=None)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    seq_length = 101
    save_path = os.path.realpath(args.save_path or cfg.get('save_path') or os.path.join(
        DATA_ROOT, 'data', args.dataset, 'data.h5'))
    assert save_path.startswith(os.path.realpath(DATA_ROOT) + os.sep), save_path

    print(f'Dataset: {args.dataset}  A-1 policy: {cfg["a1_policy"]}  p_rule: {args.p_rule}  '
          f'seed key: {args.seed_key!r}')

    env_cfg = load_config(ENV_CONFIG)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()
    policy = AgentPolicy(env.self_agent, cfg['a1_policy'], p_rule=args.p_rule, liked=args.liked)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cam_h = env.world.config.camera.agent.height
    cam_w = env.world.config.camera.agent.width
    check_disk_space(save_path, cfg['n_data'], seq_length, cam_h, cam_w)

    # layouts are drawn for the full train+test sequence so the check sets
    # (train 3) reproduce the first episodes of the full datasets
    layouts = episode_layouts(sum(FULL.values()), args.seed_key)

    with h5py.File(save_path, 'w') as h5_file:
        h5_file.attrs['dataset'] = args.dataset
        h5_file.attrs['seed_key'] = args.seed_key
        h5_file.attrs['seed_scheme'] = 'per-episode md5(<seed_key>/<split>/<n>) for random & numpy; ' \
            'coin md5(<seed_key>/<split>/<n>/a1_mix); layout md5(<seed_key>/layout)'
        h5_file.attrs['layout_seed'] = get_seed(args.seed_key + '/layout')
        h5_file.attrs['layout_mode'] = 'mix'
        h5_file.attrs['a1_policy'] = cfg['a1_policy']
        h5_file.attrs['p_rule'] = args.p_rule if cfg['a1_policy'] == 'mixed' else 0.0
        h5_file.attrs['a1_liked'] = args.liked
        h5_file.attrs['a2_policy'] = args.a2_policy
        h5_file.attrs['simulator'] = 'sim_v10'
        h5_file.attrs['other_vision_timing'] = 'pre-step (same frame as self_vision)'
        collect(env, policy, cfg['n_data'], layouts, seq_length, args.seed_key, h5_file)

    print(f'Saved: {save_path}')


if __name__ == '__main__':
    main()
