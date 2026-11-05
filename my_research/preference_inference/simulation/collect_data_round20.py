"""
Paper round 20 (paper/check/paper_instructions_round20.md; docs/paper_extra_log.md Sec.44; exploratory, outside the
pre-registration): mid-episode red/green swap test sets (Woodward-type). No training.
The first half is the episode procedure of collect_data_vdir_b.episode (= collect_data_v11_s2t, v11_s2_pref) with
A-2's rule probability p_rule (0.8 main, 0.5 descriptive); A-1 RandomAgent, A-2 'mixed' on its own view.
  * frames t = 0..T_S-1 (first half): the existing procedure, shared global RNGs (as in all existing sets);
  * at t = T_S (= 50): variants N and O swap red and green (env.set_layout(other_layout)) before frame T_S is rendered;
    variants S and R do not swap;
  * t >= T_S (second half, K = 30 A-2 actions t = 50..79, frames up to 80): A-2's p_rule = 1.0 in every variant, and
    A-2 draws from its own RNG stream (python `random` and numpy global states swapped in around pol2.step, seeded from
    '<KEY>/test/<n>/a2_post', same for all variants), so A-1's path (shared global stream) is the same in every
    variant and A-2's sleep onsets are the same in every variant;
      N  rule on A-2's actual (swapped) view, liked colour      -> new location of the liked colour
      O  rule on A-2's view re-rendered with the pre-swap layout -> old location (now the disliked colour)
      S  no swap, rule toward the liked colour
      R  no swap, rule toward the disliked colour
  * 600 episodes: layout (before the swap) x A-2 liked colour, 150 per cell, order shuffled with
    RandomState(md5(KEY + '/cells')); the same KEY (initial positions, cells) for p_rule 0.8 and 0.5.
Stored like paper_vdir_b_pref (T = 81 frames) plus `variant`, `p_rule`, `swap_t`; layout = layout before the swap,
layout_t per frame.
Usage (inside Docker, from the v11 worktree's my_research/preference_inference):
    xvfb-run --auto-servernum python simulation/collect_data_round20.py --p_rule 0.8 --variant N
-> data/data/paper_r20_swap_p80/data_N.h5 (one file per variant)
"""

import argparse
import os
import random
import sys

import h5py
import numpy as np

_SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SIM)
import collect_data_v11_s2t as S  # noqa  (sets the paths and MESA override)
import collect_data_vdir_b as VB  # noqa
from sim_v10 import layout  # noqa

KEY = 'paper_round20_swap_20261106'
N_CELL = 150
T_S = 50
K = 30
T = T_S + K + 1           # frames 0..80
VARIANTS = ('N', 'O', 'S', 'R')
OPP = {'Red': 'Green', 'Green': 'Red'}


def dataset_name(p_rule):
    return f'paper_r20_swap_p{int(round(p_rule * 100))}'


class A2Stream:
    """A-2's own RNG stream for the second half (python random + numpy global), swapped in around pol2.step."""

    def __init__(self, seed):
        self.py = random.Random(seed).getstate()
        self.np = np.random.RandomState(seed).get_state()

    def step(self, pol2, view):
        s_py, s_np = random.getstate(), np.random.get_state()
        random.setstate(self.py)
        np.random.set_state(self.np)
        try:
            return pol2.step(view)
        finally:
            self.py, self.np = random.getstate(), np.random.get_state()
            random.setstate(s_py)
            np.random.set_state(s_np)


def episode(env, pol1, pol2, n, lay0, liked, p_rule, variant):
    world = env.world
    pol2.liked = liked
    pol2.p_rule = p_rule
    env.set_layout(lay0)
    sd = S.get_seed(f'{KEY}/test/{n}')
    random.seed(sd)
    np.random.seed(sd)
    pol2.set_rng(np.random.RandomState(S.get_seed(f'{KEY}/test/{n}/a2_mix')))
    env.reset()
    buf = {k: [] for k in ('self_vision', 'other_vision') + VB.KEYS_F + VB.KEYS_I + ('layout_t',)}
    cur, a2s = lay0, None
    for t in range(T):
        if t == T_S:
            if variant in ('N', 'O'):
                cur = layout.other_layout(lay0)
                env.set_layout(cur)
            pol2.p_rule = 1.0
            if variant == 'R':
                pol2.liked = OPP[liked]
            a2s = A2Stream(S.get_seed(f'{KEY}/test/{n}/a2_post'))
        sp, op = env.self_agent.p.copy(), env.other_agent.p.copy()
        world.set_camera('self'); world.draw(env.self_agent, env.other_agent)
        sv = world.capture()
        world.set_camera('other'); world.draw(env.self_agent, env.other_agent)
        ov = world.capture()
        pv = ov
        if variant == 'O' and t >= T_S:          # A-2's policy sees the pre-swap layout
            env.set_layout(lay0)
            world.draw(env.self_agent, env.other_agent)
            pv = world.capture()
            env.set_layout(cur)
        sm, s1 = pol1.step(sv)
        om, s2 = pol2.step(pv) if a2s is None else a2s.step(pol2, pv)
        for k, v in (('self_vision', np.round(sv * 255).astype(np.uint8)),
                     ('other_vision', np.round(ov * 255).astype(np.uint8)),
                     ('self_motion', sm), ('self_position', sp), ('other_motion', om),
                     ('other_position', op), ('a1_source', s1), ('a2_source', s2),
                     ('layout_t', layout.LAYOUT_CODE[cur])):
            buf[k].append(v)
    ep = {k: np.array(v) for k, v in buf.items()}
    for k in VB.KEYS_F:
        ep[k] = ep[k].astype(np.float32)
    for k in VB.KEYS_I:
        ep[k] = ep[k].astype(np.int8)
    ep['layout_t'] = ep['layout_t'].astype(np.uint8)
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p_rule', type=float, required=True)
    ap.add_argument('--variant', required=True, choices=VARIANTS)
    ap.add_argument('--n_ep', type=int, default=0, help='0 = all 600')
    ap.add_argument('--out', default='', help='file path (default data/data/<dataset>/data_<variant>.h5)')
    args = ap.parse_args()
    cells = [(L, lk) for L in (layout.ORIGINAL, layout.SWAP_RG) for lk in ('Red', 'Green')]
    order = np.repeat(np.arange(4), N_CELL)
    np.random.RandomState(S.get_seed(KEY + '/cells')).shuffle(order)
    N = args.n_ep or len(order)
    ds_name = dataset_name(args.p_rule)
    save_path = os.path.realpath(args.out or os.path.join(S.DATA_ROOT, 'data', ds_name, f'data_{args.variant}.h5'))
    assert save_path.startswith(os.path.realpath(S.DATA_ROOT) + os.sep) and not os.path.exists(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env, pol1, pol2 = VB.make_env()
    with h5py.File(save_path, 'w') as f:
        for k, v in (('dataset', ds_name), ('seed_key', KEY), ('variant', args.variant), ('p_rule', args.p_rule),
                     ('p_rule_second_half', 1.0), ('swap_t', T_S), ('k', K), ('swap', int(args.variant in ('N', 'O'))),
                     ('a1_policy', 'random'), ('a2_policy', 'mixed on A-2 own view (O: pre-swap layout re-rendered)'),
                     ('simulator', 'sim_v10'), ('generator', 'simulation/collect_data_round20.py'),
                     ('cells_seed', S.get_seed(KEY + '/cells'))):
            f.attrs[k] = v
        g = f.create_group('test')
        ds = {k: g.create_dataset(k, (N, T, 16, 64, 3), dtype='uint8', chunks=(1, T, 16, 64, 3),
                                  compression='gzip', compression_opts=4) for k in ('self_vision', 'other_vision')}
        for k in VB.KEYS_F:
            ds[k] = g.create_dataset(k, (N, T, 2), dtype='f')
        for k in VB.KEYS_I:
            ds[k] = g.create_dataset(k, (N, T), dtype='int8')
        ds['layout_t'] = g.create_dataset('layout_t', (N, T), dtype='uint8')
        lay_ds = g.create_dataset('layout', (N,), dtype='uint8')
        lk_ds = g.create_dataset('a2_liked', (N,), dtype='uint8')    # 0 Green, 1 Red
        for n in range(N):
            lay0, liked = cells[order[n]]
            lay_ds[n] = layout.LAYOUT_CODE[lay0]
            lk_ds[n] = 0 if liked == 'Green' else 1
            ep = episode(env, pol1, pol2, n, lay0, liked, args.p_rule, args.variant)
            for k, v in ep.items():
                ds[k][n] = v
            if (n + 1) % 100 == 0:
                print(f'{n + 1}/{N}', flush=True)
    print('saved', save_path)


if __name__ == '__main__':
    main()
