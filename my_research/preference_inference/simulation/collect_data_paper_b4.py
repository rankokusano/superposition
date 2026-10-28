"""
Paper extra B4 data (docs/paper_extra_log.md Sec.29): the stage-2 test sets v11_s2t_green and the v11_s2_pref
test split with A-1 moving as in its own 80% stage-1 data (v13_s1_mixed80: AgentPolicy 'mixed', p_rule 0.8,
liked Red, coin RandomState(md5(<key>/<split>/<n>/a1_mix))) instead of the RandomAgent. A-2 is unchanged
(rule p 0.5 on its own view, same liked colour, same coin seed). Seed key, layouts, preferences and per-episode
seeds are those of collect_data_v11_s2t.main, so the first frame of every episode equals the original; A-1 and
A-2 share the global RNGs, so both paths differ from the original after the first step.
The episode loop is a copy of collect_data_v11_s2t.main; --a1 random must reproduce the original episodes bit for
bit (checked in memory against the original file for the first --n episodes; nothing is written).
Usage (inside Docker, from the v11 worktree's my_research/preference_inference):
    xvfb-run --auto-servernum python simulation/collect_data_paper_b4.py --dataset paper_s2t_green_a1p80
    xvfb-run --auto-servernum python simulation/collect_data_paper_b4.py --dataset paper_s2t_green_a1p80 --a1 random --n 5
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

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
T = 101
A1_P, A1_LIKED = 0.8, 'Red'
# dataset -> (original, seed key, liked, number of episodes the layout / preference sequences are drawn for,
#             ep_global offset of the test split)
DATASETS = {
    'paper_s2t_green_a1p80': ('v11_s2t_green', 'v11_s2t', 'Green', 300, 0),
    'paper_s2_pref_a1p80': ('v11_s2_pref', 'v11_s2', 'per_episode', 3300, 3000),
}
KEYS = ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_motion', 'other_position',
        'a1_source', 'a2_source', 'layout_t')


def get_seed(key):
    return int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--a1', default='mixed', choices=('mixed', 'random'), help='random: reproduction check only')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--check_dir', default=None, help='write a check file here instead of data/data/<dataset>')
    args = ap.parse_args()
    orig, key, liked_cfg, n_total, off = DATASETS[args.dataset]
    env_cfg = load_config(ENV_CONFIG)
    env_cfg.environment.agent.other = 'random'
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()
    world = env.world
    pol1 = AgentPolicy(env.self_agent, args.a1, p_rule=A1_P, liked=A1_LIKED)
    pol2 = AgentPolicy(env.other_agent, 'mixed', p_rule=0.5, liked='Green')
    rng_l = np.random.RandomState(get_seed(key + '/layout'))
    layouts = [layout.SWAP_RG if rng_l.rand() < 0.5 else layout.ORIGINAL for _ in range(n_total)]
    rng_p = np.random.RandomState(get_seed(key + '/pref'))
    prefs = ['Red' if rng_p.rand() < 0.5 else 'Green' for _ in range(n_total)]
    split, n_data = 'test', args.n
    buf_all = {k: [] for k in KEYS}
    lay_all, liked_all = [], []
    t0 = time.time()
    for n in range(n_data):
        lay0 = layouts[off + n]
        liked = prefs[off + n] if liked_cfg == 'per_episode' else liked_cfg
        pol2.liked = liked
        env.set_layout(lay0)
        lay_all.append(layout.LAYOUT_CODE[lay0])
        liked_all.append(0 if liked == 'Green' else 1)
        sd = get_seed(f'{key}/{split}/{n}')
        random.seed(sd)
        np.random.seed(sd)
        pol2.set_rng(np.random.RandomState(get_seed(f'{key}/{split}/{n}/a2_mix')))
        if args.a1 == 'mixed':
            pol1.set_rng(np.random.RandomState(get_seed(f'{key}/{split}/{n}/a1_mix')))
        env.reset()
        buf = {k: [] for k in KEYS}
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
        for k in KEYS:
            buf_all[k].append(np.array(buf[k]))
        if (n + 1) % 50 == 0:
            print(f'{n + 1}/{n_data} {time.time() - t0:.0f}s', flush=True)
    print(f'{n_data} episodes {time.time() - t0:.1f}s', flush=True)
    with h5py.File(os.path.join(DATA_ROOT, 'data', orig, 'data.h5'), 'r') as f:
        g = f['test']
        dt = {k: g[k].dtype for k in g}
    arr = {k: np.stack(v).astype(dt[k]) for k, v in buf_all.items()}
    arr['layout'] = np.array(lay_all, dt['layout'])
    arr['a2_liked'] = np.array(liked_all, np.uint8)
    if args.a1 == 'random':
        with h5py.File(os.path.join(DATA_ROOT, 'data', orig, 'data.h5'), 'r') as f:
            g = f['test']
            res = {k: bool(np.array_equal(arr[k], g[k][:n_data])) for k in arr if k in g}
        print('reproduce original', json.dumps(res), 'ALL', all(res.values()), flush=True)
        return
    save = os.path.join(args.check_dir, f'{args.dataset}_n{n_data}.h5') if args.check_dir else \
        os.path.join(DATA_ROOT, 'data', args.dataset, 'data.h5')
    assert os.path.realpath(save).startswith(os.path.realpath(DATA_ROOT) + os.sep)
    assert not os.path.exists(save), save
    os.makedirs(os.path.dirname(save), exist_ok=True)
    with h5py.File(save, 'w') as f:
        for k, v in (('dataset', args.dataset), ('original', orig), ('seed_key', key), ('a2_liked', liked_cfg),
                     ('a1_policy', f'mixed p_rule {A1_P} liked {A1_LIKED} (as v13_s1_mixed80)'),
                     ('a2_policy', 'mixed p_rule 0.5 on A-2 own view'), ('simulator', 'sim_v10')):
            f.attrs[k] = v
        g = f.create_group('test')
        for k, v in arr.items():
            g.create_dataset(k, data=v)
    print('saved', save, {k: round(float((arr['a1_source'] == c).mean()), 3) for k, c in (('rule', 1), ('fallback', 2), ('sleep', 3))})


if __name__ == '__main__':
    main()
