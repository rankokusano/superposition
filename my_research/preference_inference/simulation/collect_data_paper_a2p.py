"""
Paper extra A data (docs/paper_extra_log.md Sec.1.1, 2.1): the stage-2 test sets v11_s2t_green and the v11_s2_pref
test split with A-2 following its preference on p = 0.8 of awake steps (originals: 0.5), everything else paired.

collect_data_v11_s2t.py shares numpy's global RNG between A-1's and A-2's RandomAgent targets, so changing A-2's
p alone changes A-1's later targets. Here every episode is simulated in passes with the same seeds and layout:
  pass 1  the original procedure at p = 0.5 (code of collect_data_v11_s2t.main); records A-1's float64 positions,
          its returned motions and the steps at which A-1 re-drew its target;
  pass 2  A-1 REPLAYED: placed on the recorded positions (motion = the recorded one), with the random numbers
          consumed as in AgentBase.step (target re-draw at the recorded steps -> 2 numpy draws, then the python
          `random` sleep draw); A-2 runs normally with p_rule = p.
Pass 2 is run with p = 0.5 (must reproduce the original data bit for bit) and with --p (written out). Pass 1 must
also equal the original. The comparisons with the original data are made in memory for every episode and saved
to <out>.check.json; nothing is written if --check_only.
Usage (inside Docker, from the v11 worktree's my_research/preference_inference):
    xvfb-run --auto-servernum python simulation/collect_data_paper_a2p.py --dataset paper_s2t_green_a2p80
"""

import argparse
import hashlib
import json
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
from v11core.rule_agent import AgentPolicy, SRC_RANDOM, SRC_SLEEP  # noqa

DATA_ROOT = os.path.join(_PI_DIR, 'data')
ENV_CONFIG = os.path.join(_PI_DIR, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml')
T = 101
KEYS = ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_motion', 'other_position',
        'a1_source', 'a2_source', 'layout_t')
# source set -> (seed key, liked, splits of the full source collection, split used, ep_global offset)
SOURCES = {
    'v11_s2t_green': ('v11_s2t', 'Green', 300, 'test', 0),
    'v11_s2_pref': ('v11_s2', 'per_episode', 3300, 'test', 3000),
}
DATASETS = {
    'paper_s2t_green_a2p80': ('v11_s2t_green', 0.8),
    'paper_s2_pref_a2p80': ('v11_s2_pref', 0.8),
}


def get_seed(key):
    return int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16)


def episode(env, pol1, pol2, key, split, n, lay0, liked, p_rule, replay=None):
    """One episode as collect_data_v11_s2t.main (no swap). replay=None: A-1 runs its RandomAgent (pass 1, also
    returns the record for pass 2); else A-1 follows replay = (pre positions, post positions, motions, reset steps)."""
    world, a1 = env.world, env.self_agent
    pol2.liked = liked
    pol2.p_rule = p_rule
    env.set_layout(lay0)
    sd = get_seed(f'{key}/{split}/{n}')
    random.seed(sd)
    np.random.seed(sd)
    pol2.set_rng(np.random.RandomState(get_seed(f'{key}/{split}/{n}/a2_mix')))
    env.reset()
    buf = {k: [] for k in KEYS}
    rec = {'pre': [], 'post': [], 'motion': [], 'resets': []}
    orig_reset = a1.reset_target
    for t in range(T):
        if replay is not None:
            a1.p = replay[0][t].copy()
        sp, op = a1.p.copy(), env.other_agent.p.copy()
        world.set_camera('self'); world.draw(a1, env.other_agent)
        sv = world.capture()
        world.set_camera('other'); world.draw(a1, env.other_agent)
        ov = world.capture()
        if replay is None:
            called = []
            a1.reset_target = lambda: (called.append(t), orig_reset())
            sm, s1 = pol1.step(sv)
            a1.reset_target = orig_reset
            rec['pre'].append(sp); rec['post'].append(a1.p.copy()); rec['motion'].append(np.array(sm))
            rec['resets'] += called
        else:   # AgentBase.step with walk() replaced by the recorded move
            s1 = SRC_SLEEP if a1.is_sleep else SRC_RANDOM
            if a1.is_sleep:
                sm = np.array([0., 0.])
                a1.sleep_count -= 1
                if a1.sleep_count == 0:
                    a1.is_sleep = False
            else:
                sm = replay[2][t].copy()
                a1.p = replay[1][t].copy()
                if t in replay[3]:
                    orig_reset()
                if random.random() < a1.config.sleep_p:
                    a1.is_sleep = True
                    a1.sleep_count = a1.config.sleep_base + \
                        random.randrange(-a1.config.sleep_margin, a1.config.sleep_margin + 1)
        om, s2 = pol2.step(ov)
        for k, v in (('self_vision', np.round(sv * 255).astype(np.uint8)),
                     ('other_vision', np.round(ov * 255).astype(np.uint8)),
                     ('self_motion', sm), ('self_position', sp), ('other_motion', om),
                     ('other_position', op), ('a1_source', s1), ('a2_source', s2),
                     ('layout_t', layout.LAYOUT_CODE[lay0])):
            buf[k].append(v)
    ep = {k: np.array(v) for k, v in buf.items()}
    if replay is None:
        return ep, (np.array(rec['pre']), np.array(rec['post']), np.array(rec['motion']), set(rec['resets']))
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(DATASETS))
    ap.add_argument('--n_ep', type=int, default=0, help='0 = all 300')
    ap.add_argument('--save_path', default=None)
    ap.add_argument('--check_only', action='store_true')
    args = ap.parse_args()
    src, p_new = DATASETS[args.dataset]
    key, liked_cfg, n_total, split, off = SOURCES[src]
    save_path = os.path.realpath(args.save_path or os.path.join(DATA_ROOT, 'data', args.dataset, 'data.h5'))
    assert save_path.startswith(os.path.realpath(DATA_ROOT) + os.sep)
    orig = h5py.File(os.path.join(DATA_ROOT, 'data', src, 'data.h5'), 'r')[split]
    N = args.n_ep or orig['layout'].shape[0]

    env_cfg = load_config(ENV_CONFIG)
    env_cfg.environment.agent.other = 'random'
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()
    pol1 = AgentPolicy(env.self_agent, 'random')
    pol2 = AgentPolicy(env.other_agent, 'mixed', p_rule=0.5, liked='Green')
    rng_l = np.random.RandomState(get_seed(key + '/layout'))
    layouts = [layout.SWAP_RG if rng_l.rand() < 0.5 else layout.ORIGINAL for _ in range(n_total)]
    rng_p = np.random.RandomState(get_seed(key + '/pref'))
    prefs = ['Red' if rng_p.rand() < 0.5 else 'Green' for _ in range(n_total)]

    chk = {'pass1_equal_original': [], 'pass2_p05_equal_original': [], 'new_paired_keys_equal_original': []}
    paired = ('self_position', 'self_motion', 'a1_source', 'layout_t')
    out = None
    if not args.check_only:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out = h5py.File(save_path, 'w')
        for k, v in orig.file.attrs.items():
            out.attrs[k] = v
        out.attrs['dataset'] = args.dataset
        out.attrs['source'] = src
        out.attrs['a2_policy'] = f'mixed p_rule {p_new} on A-2 own view'
        out.attrs['a1_policy'] = 'random, replayed from the source data (simulation/collect_data_paper_a2p.py)'
        g = out.create_group(split)
        ds = {k: g.create_dataset(k, (N,) + orig[k].shape[1:], dtype=orig[k].dtype) for k in KEYS}
        for k in ('layout', 'a2_liked'):
            if k in orig:
                g.create_dataset(k, data=orig[k][:N])
    for n in range(N):
        lay0 = layouts[off + n]
        liked = prefs[off + n] if liked_cfg == 'per_episode' else liked_cfg
        assert layout.LAYOUT_CODE[lay0] == orig['layout'][n]
        if 'a2_liked' in orig:   # v11_s2t_green has no a2_liked (always Green)
            assert (0 if liked == 'Green' else 1) == orig['a2_liked'][n]
        o = {k: orig[k][n] for k in KEYS}
        cast = lambda e: {k: v.astype(o[k].dtype) for k, v in e.items()}  # noqa: E731 (as stored by h5py)
        e1, rp = episode(env, pol1, pol2, key, split, n, lay0, liked, 0.5)
        e1 = cast(e1)
        e2 = cast(episode(env, pol1, pol2, key, split, n, lay0, liked, 0.5, replay=rp))
        e3 = cast(episode(env, pol1, pol2, key, split, n, lay0, liked, p_new, replay=rp))
        chk['pass1_equal_original'].append(all(np.array_equal(e1[k], o[k]) for k in KEYS))
        chk['pass2_p05_equal_original'].append(all(np.array_equal(e2[k], o[k]) for k in KEYS))
        chk['new_paired_keys_equal_original'].append(all(np.array_equal(e3[k], o[k]) for k in paired))
        if out is not None:
            for k in KEYS:
                ds[k][n] = e3[k]
        if (n + 1) % 50 == 0:
            print(f'{n + 1}/{N}', {k: int(sum(v)) for k, v in chk.items()}, flush=True)
    if out is not None:
        out.close()
    summ = {k: {'n_true': int(sum(v)), 'n': len(v), 'false_eps': [i for i, x in enumerate(v) if not x]} for k, v in chk.items()}
    summ['p_new'] = p_new
    with open(save_path + ('.checkonly' if args.check_only else '') + '.check.json', 'w') as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
