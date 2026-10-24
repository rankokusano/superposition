"""
v15 stage-0 data collection (docs/v15_experiment_log.md Sec.2, 4.1).

World: v15core.world.World15 (sim_v10 field split into cubes; one colour
gray from the start of each episode; a cube turns gray when eaten).
Policies: v15core.rule_agent.PrefPolicy (v11 "mixed" rule toward the most
liked available colour with a positive value; eats within r = 1.5 of the
cube's corner).

  stage 1 (s1): A-1 follows its rank (p_rule) and eats; A-2 static (StayAgent).
  stage 2 (s2): A-1 random (RandomAgent, never eats); A-2 a RandomAgent driven
                by the rule on its OWN view, following its rank, and eats.

Per frame t (101 frames): both views and positions before the step t, the
availability (as rendered in frame t), each agent's motion/source/target,
and the colour eaten after step t (-1 none; gray from frame t + 1).
Per episode: layout, initial gray colour, the A-1/A-2 ranks and the
A-2 candidate code. Colour codes 0 Red, 1 Green, 2 Blue, 3 Cyan.

Seeding (same scheme as collect_data_v11.py): python `random` and numpy's
global RNG from md5('<key>/<split>/<n>'); coins from md5('.../a1_mix'),
md5('.../a2_mix'); layout / gray / candidate sequences from
RandomState(md5('<key>/layout' | '/gray' | '/cand')) over the full episode
order. Datasets with the same key share layouts, grays, start positions,
first random targets and sleep times (pairs differ only once a choice
differs).

Usage (inside Docker, from the v15 worktree's my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data_v15.py --dataset v15_s2_c2_p80
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

from sim_v10 import layout  # noqa
from v15core.env import Env15  # noqa
from v15core.pleasure import RANK_VALUES, pleasure_np, rank_values  # noqa
from v15core.rule_agent import PrefPolicy  # noqa
from v15core.world import COLOR_CODE, COLORS  # noqa

DATA_ROOT = os.path.join(_PI_DIR, 'data')
CHECK = os.path.join(DATA_ROOT, 'result', 'v15', 'datacheck')
T = T_DEFAULT = 101
N_EP = 1000
N_SEQ = 3300  # layout/gray/cand sequences: train 0..2999 then test 0..299 (prefix of the 1000-episode sets)
TEST_OFFSET = 3000

RANK_A1 = ('Red', 'Blue', 'Cyan', 'Green')
CAND = {
    'X': ('Red', 'Blue', 'Cyan', 'Green'),
    'Y': ('Green', 'Cyan', 'Blue', 'Red'),
    'Xp': ('Red', 'Cyan', 'Blue', 'Green'),
}
CAND_CODE = {'X': 0, 'Y': 1, 'Xp': 2}

# stage: 1 or 2; cand: fixed candidate or 'mix' (X/Y 1/2 from the cand sequence);
# layout / gray: 'seq' (key sequences) or a fixed value
DATASET_CONFIGS = {}
for _p in (80, 50):
    DATASET_CONFIGS[f'v15_s1_p{_p}'] = dict(stage=1, key='v15_s1', p=_p / 100, cand=None, layout='seq', gray='seq')
for _p in (80, 90, 50):
    DATASET_CONFIGS[f'v15_s2_c1_p{_p}'] = dict(stage=2, key='v15_s2', p=_p / 100, cand='Y', layout='seq', gray='seq')
    DATASET_CONFIGS[f'v15_s2_c2_p{_p}'] = dict(stage=2, key='v15_s2', p=_p / 100, cand='mix', layout='seq', gray='seq')
for _p in (80, 50):
    DATASET_CONFIGS[f'v15_s2_c3x_p{_p}'] = dict(stage=2, key='v15_s2c3', p=_p / 100, cand='X', layout='seq', gray='Green')
    DATASET_CONFIGS[f'v15_s2_c3xp_p{_p}'] = dict(stage=2, key='v15_s2c3', p=_p / 100, cand='Xp', layout='seq', gray='Green')
# stage-2 design proposals (docs/v15_experiment_log.md Sec.9; data only, 300 episodes, key v15_p2):
# (i) three positive colours (rank values 1, 0.5, 0.25, -1), gray every episode ('seq') or in half of them ('half');
# (ii) lower A-2 rule probability p; one longer-episode variant (T = 151).
for _g in ('seq', 'half'):
    DATASET_CONFIGS[f'v15p_i_gray{_g}_p80'] = dict(stage=2, key='v15_p2', p=0.8, cand='mix', layout='seq', gray=_g,
                                                   rank_values=(1.0, 0.5, 0.25, -1.0), n=300)
for _p in (30, 40, 50, 60, 80):
    DATASET_CONFIGS[f'v15p_ii_p{_p}'] = dict(stage=2, key='v15_p2', p=_p / 100, cand='mix', layout='seq', gray='seq', n=300)
DATASET_CONFIGS['v15p_ii_p50_T151'] = dict(stage=2, key='v15_p2', p=0.5, cand='mix', layout='seq', gray='seq', n=300, T=151)
DATASET_CONFIGS['v15_s2_genx_p80'] = dict(stage=2, key='v15_s2gen', p=0.8, cand='X', layout=layout.ORIGINAL, gray='Green')
DATASET_CONFIGS['v15_s2_geny_p80'] = dict(stage=2, key='v15_s2gen', p=0.8, cand='Y', layout=layout.SWAP_RG, gray='Red')
for _k in list(DATASET_CONFIGS):
    for _rep in ('a', 'b'):
        DATASET_CONFIGS[f'{_k}_check_{_rep}'] = dict(DATASET_CONFIGS[_k], n=20,
                                                     save_path=os.path.join(CHECK, f'{_k}_check_{_rep}.h5'))


def get_seed(key):
    return int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16)


def episode_sequences(key, n_total):
    rng_l = np.random.RandomState(get_seed(key + '/layout'))
    layouts = [layout.SWAP_RG if rng_l.rand() < 0.5 else layout.ORIGINAL for _ in range(n_total)]
    rng_g = np.random.RandomState(get_seed(key + '/gray'))
    grays = [COLORS[rng_g.randint(4)] for _ in range(n_total)]
    rng_c = np.random.RandomState(get_seed(key + '/cand'))
    cands = ['X' if rng_c.rand() < 0.5 else 'Y' for _ in range(n_total)]
    return layouts, grays, cands


def collect(cfg, n_data, split, h5, start=0):
    stage = cfg['stage']
    T = cfg.get('T', T_DEFAULT)
    vals = cfg.get('rank_values', RANK_VALUES)
    rank_values_ = lambda r: rank_values(r, vals)  # noqa: E731
    env = Env15('random', 'stay' if stage == 1 else 'random')
    env.init()
    world = env.world
    if stage == 1:
        pol1 = PrefPolicy(env.self_agent, 'mixed', p_rule=cfg['p'], rank=RANK_A1, values=rank_values_(RANK_A1))
        pol2 = None
    else:
        pol1 = PrefPolicy(env.self_agent, 'random')
        pol2 = PrefPolicy(env.other_agent, 'mixed', p_rule=cfg['p'])
    layouts, grays, cands = episode_sequences(cfg['key'], N_SEQ)
    # 'half': whether the episode has no gray at all, from its own sequence (the gray/layout draws are unchanged)
    rng_h = np.random.RandomState(get_seed(cfg['key'] + '/grayhalf'))
    gray_off = [rng_h.rand() < 0.5 for _ in range(N_SEQ)]

    g = h5.create_group(split)
    vs, ms = (n_data, T, 16, 64, 3), (n_data, T, 2)
    ds = {k: g.create_dataset(k, vs, dtype='uint8') for k in ('self_vision', 'other_vision')}
    for k in ('self_motion', 'self_position', 'other_motion', 'other_position'):
        ds[k] = g.create_dataset(k, ms, dtype='f')
    for k in ('a1_source', 'a2_source', 'a1_target', 'a2_target', 'a1_eat', 'a2_eat'):
        ds[k] = g.create_dataset(k, (n_data, T), dtype='int8')
    ds['available'] = g.create_dataset('available', (n_data, T, 4), dtype='uint8')
    for k in ('a1_pleasure', 'a2_pleasure'):
        ds[k] = g.create_dataset(k, (n_data, T, 8), dtype='f')
    ep = {k: g.create_dataset(k, (n_data,), dtype='int8') for k in ('layout', 'gray0', 'a2_cand')}
    for k in ('a1_rank', 'a2_rank'):
        ep[k] = g.create_dataset(k, (n_data, 4), dtype='int8')

    for n in range(n_data):
        ep_n = start + n
        q = ep_n if split == 'train' else TEST_OFFSET + ep_n
        lay = layouts[q] if cfg['layout'] == 'seq' else cfg['layout']
        gray = grays[q] if cfg['gray'] in ('seq', 'half') else cfg['gray']
        if cfg['gray'] == 'half' and gray_off[q]:
            gray = None
        cand = None if stage == 1 else (cands[q] if cfg['cand'] == 'mix' else cfg['cand'])
        rank2 = CAND[cand] if cand else None
        if pol2 is not None:
            pol2.set_preference(rank2, rank_values_(rank2))
        world.set_episode(lay, gray)
        ep['layout'][n] = layout.LAYOUT_CODE[lay]
        ep['gray0'][n] = COLOR_CODE[gray] if gray else -1
        ep['a2_cand'][n] = CAND_CODE[cand] if cand else -1
        ep['a1_rank'][n] = [COLOR_CODE[c] for c in RANK_A1]
        ep['a2_rank'][n] = [COLOR_CODE[c] for c in rank2] if rank2 else -1

        sd = get_seed(f'{cfg["key"]}/{split}/{ep_n}')
        random.seed(sd)
        np.random.seed(sd)
        pol1.set_rng(np.random.RandomState(get_seed(f'{cfg["key"]}/{split}/{ep_n}/a1_mix')))
        if pol2 is not None:
            pol2.set_rng(np.random.RandomState(get_seed(f'{cfg["key"]}/{split}/{ep_n}/a2_mix')))
        env.reset()
        v1 = rank_values_(RANK_A1)
        v2 = rank_values_(rank2) if rank2 else None

        buf = {k: [] for k in ds}
        for t in range(T):
            sp, op = env.self_agent.p.copy(), env.other_agent.p.copy()
            avail = world.available_vec()
            sv, ov = env.views()
            sm, s1, tg1, e1 = pol1.step(sv, world)
            if pol2 is None:
                om, s2, tg2, e2 = env.other_agent.step(), -1, None, None
            else:
                om, s2, tg2, e2 = pol2.step(ov, world)
            for e in (e1, e2):
                if e is not None:
                    world.eat(e)
            svu, ovu = np.round(sv * 255).astype(np.uint8), np.round(ov * 255).astype(np.uint8)
            for k, v in (('self_vision', svu), ('other_vision', ovu), ('self_motion', sm), ('self_position', sp),
                         ('other_motion', om), ('other_position', op), ('a1_source', s1), ('a2_source', s2),
                         ('a1_target', COLOR_CODE[tg1] if tg1 else -1), ('a2_target', COLOR_CODE[tg2] if tg2 else -1),
                         ('a1_eat', COLOR_CODE[e1] if e1 else -1), ('a2_eat', COLOR_CODE[e2] if e2 else -1),
                         ('available', avail), ('a1_pleasure', pleasure_np(svu, v1)),
                         ('a2_pleasure', pleasure_np(ovu, v2) if v2 else np.zeros(8, np.float32))):
                buf[k].append(v)
        for k, v in buf.items():
            ds[k][n] = np.array(v)
        if (n + 1) % 100 == 0:
            print(f'  {split}: {n + 1}/{n_data}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS))
    ap.add_argument('--save_path', default=None)
    ap.add_argument('--n', type=int, default=None, help='number of episodes (default: the dataset size)')
    ap.add_argument('--start', type=int, default=0, help='first episode index (chunked generation)')
    ap.add_argument('--split', default='train', choices=['train', 'test'])
    args = ap.parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    n_data = args.n or cfg.get('n', N_EP)
    save_path = os.path.realpath(args.save_path or cfg.get('save_path')
                                 or os.path.join(DATA_ROOT, 'data', args.dataset, 'data.h5'))
    assert save_path.startswith(os.path.realpath(DATA_ROOT) + os.sep), save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f'Dataset {args.dataset}: {cfg}  n={n_data} -> {save_path}', flush=True)
    with h5py.File(save_path, 'w') as f:
        for k, v in (('dataset', args.dataset), ('seed_key', cfg['key']), ('stage', cfg['stage']),
                     ('p_rule', cfg['p']), ('cand', str(cfg['cand'])), ('layout_mode', str(cfg['layout'])),
                     ('gray_mode', str(cfg['gray'])), ('r_eat', 1.5), ('rank_values', ', '.join(str(v) for v in cfg.get('rank_values', RANK_VALUES))), ('T', cfg.get('T', T_DEFAULT)),
                     ('a1_rank', ','.join(RANK_A1)), ('simulator', 'sim_v10 + v15core.World15'),
                     ('color_codes', '0 Red, 1 Green, 2 Blue, 3 Cyan'),
                     ('cand_codes', 'X 0 (R>B>C>G), Y 1 (G>C>B>R), Xp 2 (R>C>B>G)'),
                     ('source_codes', '0 random, 1 rule, 2 rule->random fallback, 3 asleep, 4 no target; -1 static A-2'),
                     ('timing', 'views/positions/available before step t; *_eat = eaten after step t (gray from t+1)')):
            f.attrs[k] = v
        f.attrs['episode_start'] = args.start
        collect(cfg, n_data, args.split, f, start=args.start)
    print('saved', save_path, flush=True)


if __name__ == '__main__':
    main()
