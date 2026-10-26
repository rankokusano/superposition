"""
v15 2(b) data (docs/v15_experiment_log.md Sec.24.5): A-1's input view with A-2 hidden (moved to (1e4, 1e4) and re-rendered
with World15; nothing else changes), as in v12 stage 2c (analyze/v12/make_hide_a2_v12.py).

Modes
  train : schedule per episode (seeded): t = 0 visible, first visible interval U{1..30}, then hidden / visible intervals
          alternate with lengths U{10..30}; in 50% of the episodes that have a first informative eat t_e (seeded draw),
          t_e-5 .. t_e+15 is also hidden. Writes a full copy of the source with `self_vision_in` and `a2_hidden` added
          (for training; the loader reads data.input_vision = self_vision_in).
  test  : hidden from t_e-5 to the end (t_e = the first informative eat; for pair sets t_e of the FIRST set is used for
          both, so the two members get the same hidden steps). Writes only `self_vision_in` and `a2_hidden` to
          data/data/<src>/hideA2_test.h5 (the evaluation reads it next to the source).
Check: on the first 5 episodes, frames re-rendered with A-2 at its true position are bit-identical to the stored view.
Parts are rendered separately (each under its own xvfb-run) and merged.

    xvfb-run ... python analyze/v15/make_hide_a2_v15.py --src v15_s2_c2_p80_half_1000t --mode train --split train --part 0 --nparts 4
    python analyze/v15/make_hide_a2_v15.py --src v15_s2_c2_p80_half_1000t --mode train --merge --nparts 4
"""

import argparse
import hashlib
import os
import shutil
import sys

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
DATA = os.path.join(_PI_DIR, 'data', 'data')
T = 101
FAR = np.array([1e4, 1e4])


def seed_of(key):
    return int(hashlib.md5(key.encode('utf-8')).hexdigest()[:8], 16)


def first_informative(eat, tgt, K=10):
    out = np.full(eat.shape[0], -1)
    for n in range(eat.shape[0]):
        for t in np.where(eat[n] >= 0)[0]:
            te = t + 1
            if 5 <= te <= T - 1 - K and tgt[n, te] >= 0:
                out[n] = te
                break
    return out


def train_schedule(src, split, te):
    n_ep = len(te)
    rng = np.random.RandomState(seed_of(f'v15_hideA2/{src}/{split}'))
    H = np.zeros((n_ep, T), np.uint8)
    force = rng.rand(n_ep) < 0.5
    for e in range(n_ep):
        t, hid = int(rng.randint(1, 31)), True
        while t < T:
            L = int(rng.randint(10, 31))
            if hid:
                H[e, t:t + L] = 1
            t += L
            hid = not hid
        if force[e] and te[e] > 0:
            H[e, max(1, te[e] - 5):min(T, te[e] + 16)] = 1
    return H


def test_schedule(te):
    H = np.zeros((len(te), T), np.uint8)
    for e, t in enumerate(te):
        if t > 0:
            H[e, max(1, t - 5):] = 1
    return H


def render_part(src, split, H, part, nparts, out_path):
    from sim_v10 import layout as L
    from v15core.env import Env15
    from v15core.world import COLORS
    g = h5py.File(os.path.join(DATA, src, 'data.h5'), 'r')[split]
    n_ep = g['self_vision'].shape[0]
    eps = np.array_split(np.arange(n_ep), nparts)[part]
    env = Env15('random', 'random')
    env.init()
    w = env.world
    w.set_camera('self')
    sv_in = np.empty((len(eps), T, 16, 64, 3), np.uint8)
    checked, mism = 0, 0
    for i, n in enumerate(eps):
        lay = L.CODE_LAYOUT[int(g['layout'][n])]
        g0 = int(g['gray0'][n])
        w.set_episode(lay, None if g0 < 0 else COLORS[g0])
        sv = g['self_vision'][n]
        sp, op, av = g['self_position'][n], g['other_position'][n], g['available'][n]
        sv_in[i] = sv
        for t in range(T):
            check = n < 5 and t % 10 == 0
            if not H[n, t] and not check:
                continue
            w.available = {c: bool(av[t, k]) for k, c in enumerate(COLORS)}
            w._update_cubes()
            env.self_agent.p = sp[t].astype(np.float64)
            if check:
                env.other_agent.p = op[t].astype(np.float64)
                w.draw(env.self_agent, env.other_agent)
                v = np.round(w.capture() * 255).astype(np.uint8)
                checked += 1
                mism += int(not np.array_equal(v, sv[t]))
            if H[n, t]:
                env.other_agent.p = FAR.copy()
                w.draw(env.self_agent, env.other_agent)
                sv_in[i, t] = np.round(w.capture() * 255).astype(np.uint8)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('self_vision_in', data=sv_in)
        f.attrs['episodes'] = eps
        f.attrs['rerender_check'] = f'{checked} frames, {mism} mismatches'
    print(out_path, f'rerender check: {checked} frames, {mism} mismatches', flush=True)
    if mism:
        raise SystemExit('re-render mismatch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--mode', choices=['train', 'test'], required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--te_from', default=None, help='test mode: dataset whose first informative eat defines t_e')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    a = ap.parse_args()
    splits = ['train', 'test'] if a.mode == 'train' else [a.split]
    pdir = os.path.join(DATA, a.src, f'hideA2_{a.mode}_parts')
    os.makedirs(pdir, exist_ok=True)

    def sched(split):
        src_te = a.te_from or a.src
        with h5py.File(os.path.join(DATA, src_te, 'data.h5'), 'r') as f:
            te = first_informative(f[split]['a2_eat'][()], f[split]['a2_target'][()])
        return train_schedule(a.src, split, te) if a.mode == 'train' else test_schedule(te)

    if not a.merge:
        for split in splits:
            render_part(a.src, split, sched(split), a.part, a.nparts, os.path.join(pdir, f'{split}_{a.part}.h5'))
        return
    if a.mode == 'train':
        dst = os.path.join(DATA, a.src + '_hideA2', 'data.h5')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(DATA, a.src, 'data.h5'), dst)
    else:
        dst = os.path.join(DATA, a.src, 'hideA2_test.h5')
    with h5py.File(dst, 'a' if a.mode == 'train' else 'w') as fo:
        for split in splits:
            H = sched(split)
            parts = [h5py.File(os.path.join(pdir, f'{split}_{p}.h5'), 'r') for p in range(a.nparts)]
            sv_in = np.concatenate([p['self_vision_in'][()] for p in parts])
            for p in parts:
                print(split, p.attrs['rerender_check'], flush=True)
            fo.create_dataset(f'{split}/self_vision_in', data=sv_in)
            fo.create_dataset(f'{split}/a2_hidden', data=H)
            print(split, 'hidden fraction', float(H[:, 1:].mean()), 'episodes', len(H), flush=True)
            # hidden steps differ from the original only where A-2 was visible; visible steps are copies
            with h5py.File(os.path.join(DATA, a.src, 'data.h5'), 'r') as fs:
                assert np.array_equal(sv_in[H == 0], fs[split]['self_vision'][()][H == 0]), 'visible steps changed'
    shutil.rmtree(pdir)
    print('saved', dst, flush=True)


if __name__ == '__main__':
    main()
