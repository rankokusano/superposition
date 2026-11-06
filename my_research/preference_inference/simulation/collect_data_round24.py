"""
Paper round 24 (paper/check/paper_instrctions_round24.md; docs/paper_extra_log.md Sec.48; exploratory, no training):
stage-II data procedure (collect_data_v11_s2t.py, dataset v11_s2_pref: A-1 RandomAgent, A-2 'mixed' rule agent on
A-2's OWN view, liked colour Red / Green per episode with prob 1/2, layout per episode with prob 1/2, T = 101 frames)
with a mid-episode red/green swap as in round 20 (env.set_layout(other_layout) before frame swap_t is rendered).
Options:
  --p_rule        A-2's rule probability for the whole episode (stage II: 0.5; round 20 first half: 0.8)
  --swap_t        'none', a frame number (e.g. 50) or 'rand<lo>-<hi>' (uniform integer in [lo, hi], per episode)
  --p_swap        probability that an episode has the swap (per episode, own RNG stream)
  --no_vision     do not store the views (A-1's view is then not rendered; A-1's RandomAgent does not use it, so
                  positions are identical with and without views; A-2's view is always rendered for its policy)
Per-episode seeds as v11_s2t ('<key>/<split>/<n>', '/a2_mix'); layout, liked colour, swap and swap time from their own
RandomState streams ('<key>/layout', '/pref', '/swap', '/swap_t'). A-2's rule follows its CURRENT view, so after a swap
it heads for the liked colour's new place (AgentPolicy.step -> color_direction(vision, liked)).
Usage (inside Docker): xvfb-run --auto-servernum python simulation/collect_data_round24.py --p_rule 0.5 --swap_t 50 \
    --p_swap 1.0 --n_ep 600 --part 0 --n_parts 2 --no_vision --out <file under data/>
"""

import argparse
import os
import random
import sys

import h5py
import numpy as np

_SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SIM)
import collect_data_v11_s2t as S  # noqa  (paths, MESA override, get_seed)
from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa
from v11core.rule_agent import AgentPolicy  # noqa

KEY = 'paper_round24_swap_20261107'
T = 101
KEYS_F = ('self_motion', 'self_position', 'other_motion', 'other_position')
KEYS_I = ('a1_source', 'a2_source')


def make_env():
    env_cfg = load_config(S.ENV_CONFIG)
    env_cfg.environment.agent.other = 'random'
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()
    return env, AgentPolicy(env.self_agent, 'random'), AgentPolicy(env.other_agent, 'mixed', p_rule=0.5, liked='Green')


def plan(key, n_total, p_swap, swap_t):
    rl = np.random.RandomState(S.get_seed(key + '/layout'))
    rp = np.random.RandomState(S.get_seed(key + '/pref'))
    rs = np.random.RandomState(S.get_seed(key + '/swap'))
    rt = np.random.RandomState(S.get_seed(key + '/swap_t'))
    lays = [layout.SWAP_RG if rl.rand() < 0.5 else layout.ORIGINAL for _ in range(n_total)]
    prefs = ['Red' if rp.rand() < 0.5 else 'Green' for _ in range(n_total)]
    sw = [bool(rs.rand() < p_swap) for _ in range(n_total)]
    if swap_t == 'none':
        ts = [-1] * n_total
    elif swap_t.startswith('rand'):
        lo, hi = (int(x) for x in swap_t[4:].split('-'))
        ts = [int(rt.randint(lo, hi + 1)) for _ in range(n_total)]
    else:
        ts = [int(swap_t)] * n_total
    ts = [t if s and t >= 0 else -1 for t, s in zip(ts, sw)]
    return lays, prefs, ts


def episode(env, pol1, pol2, key, split, n, lay0, liked, p_rule, ts, vision):
    world = env.world
    pol2.liked = liked
    pol2.p_rule = p_rule
    env.set_layout(lay0)
    sd = S.get_seed(f'{key}/{split}/{n}')
    random.seed(sd)
    np.random.seed(sd)
    pol2.set_rng(np.random.RandomState(S.get_seed(f'{key}/{split}/{n}/a2_mix')))
    env.reset()
    buf = {k: [] for k in KEYS_F + KEYS_I + ('layout_t',) + (('self_vision', 'other_vision') if vision else ())}
    cur = lay0
    for t in range(T):
        if t == ts:
            cur = layout.other_layout(lay0)
            env.set_layout(cur)
        sp, op = env.self_agent.p.copy(), env.other_agent.p.copy()
        sv = None
        if vision:
            world.set_camera('self'); world.draw(env.self_agent, env.other_agent)
            sv = world.capture()
        world.set_camera('other'); world.draw(env.self_agent, env.other_agent)
        ov = world.capture()
        sm, s1 = pol1.step(sv)
        om, s2 = pol2.step(ov)
        for k, v in (('self_motion', sm), ('self_position', sp), ('other_motion', om), ('other_position', op),
                     ('a1_source', s1), ('a2_source', s2), ('layout_t', layout.LAYOUT_CODE[cur])):
            buf[k].append(v)
        if vision:
            buf['self_vision'].append(np.round(sv * 255).astype(np.uint8))
            buf['other_vision'].append(np.round(ov * 255).astype(np.uint8))
    return {k: np.array(v) for k, v in buf.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', default=KEY)
    ap.add_argument('--split', default='train')
    ap.add_argument('--p_rule', type=float, required=True)
    ap.add_argument('--swap_t', required=True)
    ap.add_argument('--p_swap', type=float, default=1.0)
    ap.add_argument('--n_ep', type=int, required=True)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--n_parts', type=int, default=1)
    ap.add_argument('--no_vision', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = os.path.realpath(a.out)
    assert out.startswith(os.path.realpath(S.DATA_ROOT) + os.sep) and not os.path.exists(out), out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lays, prefs, ts = plan(a.key, a.n_ep, a.p_swap, a.swap_t)
    idx = np.array_split(np.arange(a.n_ep), a.n_parts)[a.part]
    env, pol1, pol2 = make_env()
    vision = not a.no_vision
    with h5py.File(out, 'w') as f:
        for k, v in (('key', a.key), ('split', a.split), ('p_rule', a.p_rule), ('swap_t', a.swap_t), ('p_swap', a.p_swap),
                     ('n_ep_total', a.n_ep), ('generator', 'simulation/collect_data_round24.py')):
            f.attrs[k] = v
        g = {k: f.create_dataset(k, (len(idx), T, 2), dtype='f') for k in KEYS_F}
        g.update({k: f.create_dataset(k, (len(idx), T), dtype='int8') for k in KEYS_I})
        g['layout_t'] = f.create_dataset('layout_t', (len(idx), T), dtype='uint8')
        if vision:
            for k in ('self_vision', 'other_vision'):
                g[k] = f.create_dataset(k, (len(idx), T, 16, 64, 3), dtype='uint8', chunks=(1, T, 16, 64, 3),
                                        compression='gzip', compression_opts=4)
        f.create_dataset('idx', data=idx)
        f.create_dataset('layout', data=np.array([layout.LAYOUT_CODE[lays[n]] for n in idx], np.uint8))
        f.create_dataset('a2_liked', data=np.array([1 if prefs[n] == 'Red' else 0 for n in idx], np.uint8))
        f.create_dataset('swap_frame', data=np.array([ts[n] for n in idx], np.int16))
        for i, n in enumerate(idx):
            ep = episode(env, pol1, pol2, a.key, a.split, int(n), lays[n], prefs[n], a.p_rule, ts[n], vision)
            for k, v in ep.items():
                g[k][i] = v
            if (i + 1) % 100 == 0:
                print(f'{i + 1}/{len(idx)}', flush=True)
    print('saved', out, flush=True)


if __name__ == '__main__':
    main()
