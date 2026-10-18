"""
v11 test data for the stage-2 checks T2 and T5 (docs/v11_experiment_log.md
Sec.7.37). A copy of collect_data_v11.py with:

  * A-1: RandomAgent (the paper's random walk) in every set;
  * A-2: a RandomAgent driven by the stage-1 visual rule on A-2's OWN view
    (v11core.rule_agent.AgentPolicy 'mixed', p_rule 0.5, liked colour given per
    set; a2_source recorded per step, same codes as a1_source);
  * sets (one shared seed key 'v11_s2t' -> same layout sequence and the same
    initial A-1/A-2 positions and first targets per episode in every set):
      v11_s2t_green      A-2 likes Green, no swap
      v11_s2t_green_swap A-2 likes Green, red and green swapped at t = 50 (T2)
      v11_s2t_red        A-2 likes Red, no swap (T5)
  * per-step layout `layout_t` (n, 101) besides the per-episode starting
    `layout`; the swap is applied before rendering frame t = SWAP_T.
Because A-1's and A-2's RandomAgents share the global RNGs, A-1's path is
identical across sets only until A-2's moves first differ (recorded caveat).
Usage (inside Docker, from the v11 worktree's my_research/preference_inference):
    xvfb-run --auto-servernum python simulation/collect_data_v11_s2t.py --dataset v11_s2t_green
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
ENV_CONFIG = os.path.join(_PI_DIR, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml')
SEED_KEY = 'v11_s2t'
SWAP_T = 50
N_TEST = 300
GATE = os.path.join(DATA_ROOT, 'result', 'v11', 'gate_s2t')

DATASET_CONFIGS = {
    'v11_s2t_green': {'liked': 'Green', 'swap': False},
    'v11_s2t_green_swap': {'liked': 'Green', 'swap': True},
    'v11_s2t_red': {'liked': 'Red', 'swap': False},
}
for _k in list(DATASET_CONFIGS):
    for _rep in ('a', 'b'):
        DATASET_CONFIGS[f'{_k}_check_{_rep}'] = dict(DATASET_CONFIGS[_k], n=3,
                                                     save_path=os.path.join(GATE, f'{_k}_check_{_rep}.h5'))


def get_seed(key):
    return int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS))
    args = ap.parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    n_data = cfg.get('n', N_TEST)
    save_path = os.path.realpath(cfg.get('save_path') or os.path.join(DATA_ROOT, 'data', args.dataset, 'data.h5'))
    assert save_path.startswith(os.path.realpath(DATA_ROOT) + os.sep)
    env_cfg = load_config(ENV_CONFIG)
    env_cfg.environment.agent.other = 'random'
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()
    world = env.world
    pol1 = AgentPolicy(env.self_agent, 'random')
    pol2 = AgentPolicy(env.other_agent, 'mixed', p_rule=0.5, liked=cfg['liked'])
    rng_l = np.random.RandomState(get_seed(SEED_KEY + '/layout'))
    layouts = [layout.SWAP_RG if rng_l.rand() < 0.5 else layout.ORIGINAL for _ in range(N_TEST)]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T = 101
    with h5py.File(save_path, 'w') as f:
        for k, v in (('dataset', args.dataset), ('seed_key', SEED_KEY), ('a2_liked', cfg['liked']),
                     ('swap', int(cfg['swap'])), ('swap_t', SWAP_T), ('a1_policy', 'random'),
                     ('a2_policy', 'mixed p_rule 0.5 on A-2 own view'), ('simulator', 'sim_v10')):
            f.attrs[k] = v
        g = f.create_group('test')
        vs = (n_data, T, 16, 64, 3)
        ds = {k: g.create_dataset(k, vs, dtype='uint8') for k in ('self_vision', 'other_vision')}
        for k in ('self_motion', 'self_position', 'other_motion', 'other_position'):
            ds[k] = g.create_dataset(k, (n_data, T, 2), dtype='f')
        for k in ('a1_source', 'a2_source'):
            ds[k] = g.create_dataset(k, (n_data, T), dtype='int8')
        ds['layout'] = g.create_dataset('layout', (n_data,), dtype='uint8')
        ds['layout_t'] = g.create_dataset('layout_t', (n_data, T), dtype='uint8')
        for n in range(n_data):
            lay0 = layouts[n]
            env.set_layout(lay0)
            ds['layout'][n] = layout.LAYOUT_CODE[lay0]
            s = get_seed(f'{SEED_KEY}/test/{n}')
            random.seed(s)
            np.random.seed(s)
            pol2.set_rng(np.random.RandomState(get_seed(f'{SEED_KEY}/test/{n}/a2_mix')))
            env.reset()
            buf = {k: [] for k in ds if k != 'layout'}
            cur = lay0
            for t in range(T):
                if cfg['swap'] and t == SWAP_T:
                    cur = layout.other_layout(lay0)
                    env.set_layout(cur)
                sp, op = env.self_agent.p.copy(), env.other_agent.p.copy()
                world.set_camera('self'); world.draw(env.self_agent, env.other_agent)
                sv = world.capture()
                world.set_camera('other'); world.draw(env.self_agent, env.other_agent)
                ov = world.capture()
                sm, s1 = pol1.step(sv)
                om, s2 = pol2.step(ov)
                for k, v in (('self_vision', np.round(sv * 255).astype(np.uint8)),
                             ('other_vision', np.round(ov * 255).astype(np.uint8)),
                             ('self_motion', sm), ('self_position', sp), ('other_motion', om),
                             ('other_position', op), ('a1_source', s1), ('a2_source', s2),
                             ('layout_t', layout.LAYOUT_CODE[cur])):
                    buf[k].append(v)
            for k, v in buf.items():
                ds[k][n] = np.array(v)
            if (n + 1) % 50 == 0:
                print(f'{n + 1}/{n_data}', flush=True)
    print('saved', save_path)


if __name__ == '__main__':
    main()
