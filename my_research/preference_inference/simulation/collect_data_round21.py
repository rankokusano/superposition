"""
Paper round 21 (paper/check/paper_instructions_round21.md; docs/paper_extra_log.md Sec.45; exploratory, outside the
pre-registration): mid-episode red/green swap with a static A-2, for the input-freeze tests of experiment 1
(G3 self, G5b other). No training.
Episode procedure = collect_data_v11.collect (v11_s1_random: env config self_random_other_stay, A-1 AgentPolicy
'random', A-2 StayAgent at a random static position), with
  * a new seed key: python `random` and numpy global re-seeded per episode from md5('<KEY>/test/<n>'), A-1 coin
    RandomState(md5('<KEY>/test/<n>/a1_mix')) (unused by 'random', kept as in v11);
  * 600 episodes, layout before the swap balanced (300 ORIGINAL, 300 SWAP_RG), order shuffled with
    RandomState(md5(KEY + '/cells'));
  * T = 41 frames (0..40);
  * variant 'swap': at t = T_S (= 20), before frame T_S is rendered, env.set_layout(other_layout(layout)) (red and
    green exchange places, as round 20); variant 'noswap': no change. A-1's random walk does not read its view, so
    the two variants are meant to be bit-identical before T_S and to share A-1's path throughout (checked, Sec.1.2).
Stored like v11_s1_random (self/other vision uint8 gzip, motions, positions, a1_source, layout) plus layout_t per
frame and attrs variant / swap_t.
Usage (inside Docker, from the v11 worktree's my_research/preference_inference):
    xvfb-run --auto-servernum python simulation/collect_data_round21.py --variant swap
-> data/data/paper_r21_swap/data_<variant>.h5
"""

import argparse
import os
import random
import sys

import h5py
import numpy as np

_SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SIM)
import collect_data_v11 as V  # noqa  (paths, MESA override, get_seed, env config)
from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa
from v11core.rule_agent import AgentPolicy  # noqa

KEY = 'paper_round21_swap_20261106'
N_EP = 600
T_S = 20
T = 41
VARIANTS = ('swap', 'noswap')
DS_NAME = 'paper_r21_swap'


def make_env():
    env = creator.create_environment(load_config(V.ENV_CONFIG).environment)
    env.init()
    env.off_display()
    policy = AgentPolicy(env.self_agent, 'random', p_rule=0.0, liked='Red')
    return env, policy


def episode_layouts():
    order = np.repeat([layout.ORIGINAL, layout.SWAP_RG], N_EP // 2)
    np.random.RandomState(V.get_seed(KEY + '/cells')).shuffle(order)
    return [str(x) for x in order]


def episode(env, policy, n, lay0, variant):
    world = env.world
    env.set_layout(lay0)
    sd = V.get_seed(f'{KEY}/test/{n}')
    random.seed(sd)
    np.random.seed(sd)
    policy.set_rng(np.random.RandomState(V.get_seed(f'{KEY}/test/{n}/a1_mix')))
    env.reset()
    buf = {k: [] for k in ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_motion',
                           'other_position', 'a1_source', 'layout_t')}
    cur = lay0
    for t in range(T):
        if t == T_S and variant == 'swap':
            cur = layout.other_layout(lay0)
            env.set_layout(cur)
        sp, op = env.self_agent.p.copy(), env.other_agent.p.copy()
        world.set_camera('self'); world.draw(env.self_agent, env.other_agent)
        sv = world.capture()
        world.set_camera('other'); world.draw(env.self_agent, env.other_agent)
        ov = world.capture()
        sm, src = policy.step(sv)
        om = env.other_agent.step()
        for k, v in (('self_vision', V.vision_to_uint8(sv)), ('other_vision', V.vision_to_uint8(ov)),
                     ('self_motion', sm), ('self_position', sp), ('other_motion', om), ('other_position', op),
                     ('a1_source', src), ('layout_t', layout.LAYOUT_CODE[cur])):
            buf[k].append(v)
    ep = {k: np.array(v) for k, v in buf.items()}
    for k in ('self_motion', 'self_position', 'other_motion', 'other_position'):
        ep[k] = ep[k].astype(np.float32)
    ep['a1_source'] = ep['a1_source'].astype(np.int8)
    ep['layout_t'] = ep['layout_t'].astype(np.uint8)
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=VARIANTS)
    ap.add_argument('--n_ep', type=int, default=0, help='0 = all 600')
    ap.add_argument('--out', default='', help='file path (default data/data/paper_r21_swap/data_<variant>.h5)')
    args = ap.parse_args()
    lays = episode_layouts()
    N = args.n_ep or N_EP
    save_path = os.path.realpath(args.out or os.path.join(V.DATA_ROOT, 'data', DS_NAME, f'data_{args.variant}.h5'))
    assert save_path.startswith(os.path.realpath(V.DATA_ROOT) + os.sep) and not os.path.exists(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env, policy = make_env()
    with h5py.File(save_path, 'w') as f:
        for k, v in (('dataset', DS_NAME), ('seed_key', KEY), ('variant', args.variant), ('swap_t', T_S),
                     ('swap', int(args.variant == 'swap')), ('a1_policy', 'random'), ('a2_policy', 'stay'),
                     ('a1_liked', 'Red'), ('simulator', 'sim_v10'), ('env_config', 'self_random_other_stay'),
                     ('generator', 'simulation/collect_data_round21.py'), ('cells_seed', V.get_seed(KEY + '/cells'))):
            f.attrs[k] = v
        g = f.create_group('test')
        ds = {k: g.create_dataset(k, (N, T, 16, 64, 3), dtype='uint8', chunks=(1, T, 16, 64, 3),
                                  compression='gzip', compression_opts=4) for k in ('self_vision', 'other_vision')}
        for k in ('self_motion', 'self_position', 'other_motion', 'other_position'):
            ds[k] = g.create_dataset(k, (N, T, 2), dtype='f')
        ds['a1_source'] = g.create_dataset('a1_source', (N, T), dtype='int8')
        ds['layout_t'] = g.create_dataset('layout_t', (N, T), dtype='uint8')
        lay_ds = g.create_dataset('layout', (N,), dtype='uint8')
        for n in range(N):
            lay_ds[n] = layout.LAYOUT_CODE[lays[n]]
            ep = episode(env, policy, n, lays[n], args.variant)
            for k, v in ep.items():
                ds[k][n] = v
            if (n + 1) % 100 == 0:
                print(f'{n + 1}/{N}', flush=True)
    print('saved', save_path)


if __name__ == '__main__':
    main()
