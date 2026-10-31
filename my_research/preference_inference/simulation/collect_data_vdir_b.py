"""
Paper round 13 rev2 (V-dir stage B, docs/paper_extra_log.md Sec.37): a new preference-mixed evaluation set.
The episode procedure is that of collect_data_v11_s2t.py (v11_s2_pref, unchanged, imported): A-1 RandomAgent,
A-2 'mixed' p_rule 0.5 on its own view, 101 frames, no swap, initial positions from env.reset() (uniform, not
stratified, not re-placed). Differences:
  * a new seed key (KEY below; used by no other data set);
  * layout x liked colour: exactly 300 episodes in each of the 4 cells, in an order shuffled with
    RandomState(md5(KEY + '/cells')) (instead of independent coin flips).
--check_existing regenerates the first --n_ep episodes of v11_s2_pref/test with this code (key 'v11_s2', its
layout / preference sequences) and compares them with the stored data bit for bit (nothing is written but a json).
Usage (inside Docker, from the v11 worktree's my_research/preference_inference):
    xvfb-run --auto-servernum python simulation/collect_data_vdir_b.py              (writes the set)
    xvfb-run --auto-servernum python simulation/collect_data_vdir_b.py --check_existing --n_ep 5
"""

import argparse
import json
import os
import random
import sys

import h5py
import numpy as np

_SIM = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SIM)
import collect_data_v11_s2t as S  # noqa  (sets the paths and MESA override)
from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa
from v11core.rule_agent import AgentPolicy  # noqa

KEY = 'paper_vdir_stageB_pref_20261031'
DATASET = 'paper_vdir_b_pref'
N_CELL = 300
T = 101
KEYS_F = ('self_motion', 'self_position', 'other_motion', 'other_position')
KEYS_I = ('a1_source', 'a2_source')


def make_env():
    env_cfg = load_config(S.ENV_CONFIG)
    env_cfg.environment.agent.other = 'random'
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()
    pol1 = AgentPolicy(env.self_agent, 'random')
    pol2 = AgentPolicy(env.other_agent, 'mixed', p_rule=0.5, liked='Green')
    return env, pol1, pol2


def episode(env, pol1, pol2, key, split, n, lay0, liked):
    """collect_data_v11_s2t.main's inner loop (no swap)."""
    world = env.world
    pol2.liked = liked
    env.set_layout(lay0)
    sd = S.get_seed(f'{key}/{split}/{n}')
    random.seed(sd)
    np.random.seed(sd)
    pol2.set_rng(np.random.RandomState(S.get_seed(f'{key}/{split}/{n}/a2_mix')))
    env.reset()
    buf = {k: [] for k in ('self_vision', 'other_vision') + KEYS_F + KEYS_I + ('layout_t',)}
    for t in range(T):
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
                     ('layout_t', layout.LAYOUT_CODE[lay0])):
            buf[k].append(v)
    ep = {k: np.array(v) for k, v in buf.items()}
    for k in KEYS_F:
        ep[k] = ep[k].astype(np.float32)        # stored as 'f' in the h5 files
    for k in KEYS_I:
        ep[k] = ep[k].astype(np.int8)
    ep['layout_t'] = ep['layout_t'].astype(np.uint8)
    return ep


def check_existing(n_ep):
    env, pol1, pol2 = make_env()
    key, n_total, off = 'v11_s2', 3300, 3000
    rng_l = np.random.RandomState(S.get_seed(key + '/layout'))
    layouts = [layout.SWAP_RG if rng_l.rand() < 0.5 else layout.ORIGINAL for _ in range(n_total)]
    rng_p = np.random.RandomState(S.get_seed(key + '/pref'))
    prefs = ['Red' if rng_p.rand() < 0.5 else 'Green' for _ in range(n_total)]
    g = h5py.File(os.path.join(S.DATA_ROOT, 'data', 'v11_s2_pref', 'data.h5'), 'r')['test']
    res = []
    for n in range(n_ep):
        ep = episode(env, pol1, pol2, key, 'test', n, layouts[off + n], prefs[off + n])
        res.append({k: bool(np.array_equal(ep[k], g[k][n])) for k in ep})
    out = {'n_ep': n_ep, 'all_equal': all(all(r.values()) for r in res), 'per_episode': res}
    p = os.path.join(S.DATA_ROOT, 'result', 'paper_extra', 'Vdir_B', 'data_check_existing.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, 'w'), indent=1)
    print(json.dumps({k: out[k] for k in ('n_ep', 'all_equal')}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check_existing', action='store_true')
    ap.add_argument('--n_ep', type=int, default=0, help='0 = all 1200 (with --check_existing: episodes to compare)')
    args = ap.parse_args()
    if args.check_existing:
        return check_existing(args.n_ep or 5)
    cells = [(L, lk) for L in (layout.ORIGINAL, layout.SWAP_RG) for lk in ('Red', 'Green')]
    order = np.repeat(np.arange(4), N_CELL)
    np.random.RandomState(S.get_seed(KEY + '/cells')).shuffle(order)
    N = args.n_ep or len(order)
    save_path = os.path.realpath(os.path.join(S.DATA_ROOT, 'data', DATASET, 'data.h5'))
    assert save_path.startswith(os.path.realpath(S.DATA_ROOT) + os.sep) and not os.path.exists(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    env, pol1, pol2 = make_env()
    with h5py.File(save_path, 'w') as f:
        for k, v in (('dataset', DATASET), ('seed_key', KEY), ('a2_liked', 'per_episode, 300 per layout x liked cell'),
                     ('swap', 0), ('a1_policy', 'random'), ('a2_policy', 'mixed p_rule 0.5 on A-2 own view'),
                     ('simulator', 'sim_v10'), ('generator', 'simulation/collect_data_vdir_b.py'),
                     ('cells_seed', S.get_seed(KEY + '/cells'))):
            f.attrs[k] = v
        g = f.create_group('test')
        ds = {k: g.create_dataset(k, (N, T, 16, 64, 3), dtype='uint8') for k in ('self_vision', 'other_vision')}
        for k in KEYS_F:
            ds[k] = g.create_dataset(k, (N, T, 2), dtype='f')
        for k in KEYS_I:
            ds[k] = g.create_dataset(k, (N, T), dtype='int8')
        ds['layout_t'] = g.create_dataset('layout_t', (N, T), dtype='uint8')
        lay_ds = g.create_dataset('layout', (N,), dtype='uint8')
        lk_ds = g.create_dataset('a2_liked', (N,), dtype='uint8')    # 0 Green, 1 Red
        for n in range(N):
            lay0, liked = cells[order[n]]
            lay_ds[n] = layout.LAYOUT_CODE[lay0]
            lk_ds[n] = 0 if liked == 'Green' else 1
            ep = episode(env, pol1, pol2, KEY, 'test', n, lay0, liked)
            for k, v in ep.items():
                ds[k][n] = v
            if (n + 1) % 100 == 0:
                print(f'{n + 1}/{N}', flush=True)
    print('saved', save_path)


if __name__ == '__main__':
    main()
