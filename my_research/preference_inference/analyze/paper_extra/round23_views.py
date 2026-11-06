"""
Paper round 23 (docs/paper_extra_log.md Sec.47): reference views for the feature read-out (model independent,
CPU/xvfb, as round22_views.py). A-1 is drawn from its real position of each frame.
  --mode r20  (Sec.2.1, round 20 data paper_r20_swap_p50): frames 54..80 (the frames predicted at steps 53..79, the
              round-20 evaluation interval); A-2 at p50 + d unit(L - p50), p50 = A-2's position at frame 50 (the swap),
              d = -10..20, L in {X, Y} (A-2's liked colour in the pre-swap / swapped layout); both layouts.
              views[n, frame, target, layout, d]; valid[n, target, d] (|q| <= 10; invalid views are not drawn).
              Re-render check: frames 54 and 80 of N (swapped layout) and S (pre-swap layout) with A-2 at its real
              position equal the data frames.
  --mode dir  (Sec.2.2, round 21 data_noswap): frames 24..33; A-2 at p2 + d BIN_UNIT[b], d = -5..5, b = the range of
              class E / B / R (Round23/check/ranges.npz); pre-swap layout. views[n, frame, class, d]; valid[n, class, d]
              (also False when the class has no range). Re-render check: d = 0 equals the data frame.
Usage: xvfb-run --auto-servernum python analyze/paper_extra/round23_views.py --mode r20|dir --part i --n_parts n
-> data/result/paper_extra/Round23/views_<mode>/views_part<i>of<n>.h5
"""

import argparse
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
from sim_v10 import creator, layout as L  # noqa
from sim_v10.util import load_config  # noqa
from v11core.pleasure import BIN_UNIT  # noqa

DATA = os.path.join(_PI, 'data', 'data')
OUT = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round23')
R20_FRAMES = np.arange(54, 81)
R20_D = np.arange(-10, 21)
DIR_FRAMES = np.arange(24, 34)
DIR_D = np.arange(-5, 6)
CLASSES = ('E', 'B', 'R')


def render(env, sp, op):
    env.self_agent.p = np.asarray(sp, dtype=np.float64)
    env.other_agent.p = np.asarray(op, dtype=np.float64)
    env.world.set_camera('self')
    env.world.draw(env.self_agent, env.other_agent)
    return np.round(env.world.capture() * 255).astype(np.uint8)


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def run_r20(env, idx, f):
    G = {v: h5py.File(os.path.join(DATA, 'paper_r20_swap_p50', f'data_{v}.h5'), 'r')['test'] for v in 'NS'}
    g = G['N']
    pos, opos, lay, lk = g['self_position'][()], g['other_position'][()], g['layout'][()], g['a2_liked'][()]
    nf, nd = len(R20_FRAMES), len(R20_D)
    vv = f.create_dataset('views', (len(idx), nf, 2, 2, nd, 16, 64, 3), dtype='uint8',
                          chunks=(1, nf, 2, 2, nd, 16, 64, 3), compression='gzip', compression_opts=4)
    vok = f.create_dataset('valid', (len(idx), 2, nd), dtype='bool')
    nbad = 0
    for i, n in enumerate(idx):
        lay0 = L.CODE_LAYOUT[int(lay[n])]
        lay1 = L.other_layout(lay0)
        col = 'Red' if lk[n] == 1 else 'Green'
        p50 = opos[n, 50].astype(np.float64)
        us = [unit(np.asarray(L.LANDMARKS[ly][col], np.float64) - p50) for ly in (lay0, lay1)]   # X, Y
        ok = np.array([[not np.any(np.abs(p50 + d * u) > 10) for d in R20_D] for u in us])
        buf = np.zeros((nf, 2, 2, nd, 16, 64, 3), np.uint8)
        for li, ly in enumerate((lay0, lay1)):
            env.set_layout(ly)
            for fi, fr in enumerate(R20_FRAMES):
                sp = pos[n, fr].astype(np.float64)
                for ti, u in enumerate(us):
                    for di, d in enumerate(R20_D):
                        if ok[ti, di]:
                            buf[fi, ti, li, di] = render(env, sp, p50 + d * u)
            # re-render check: S uses the pre-swap layout, N the swapped one, from frame 50 on
            v = 'S' if li == 0 else 'N'
            for fr in (54, 80):
                img = render(env, pos[n, fr], G[v]['other_position'][n, fr])
                nbad += int(not np.array_equal(img, G[v]['self_vision'][n, fr]))
        vv[i], vok[i] = buf, ok
        if (i + 1) % 10 == 0:
            print(f'{i + 1}/{len(idx)}', flush=True)
    return nbad


def run_dir(env, idx, f):
    g = h5py.File(os.path.join(DATA, 'paper_r21_swap', 'data_noswap.h5'), 'r')['test']
    pos, opos, lay = g['self_position'][()], g['other_position'][()], g['layout'][()]
    rg = np.load(os.path.join(OUT, 'check', 'ranges.npz'))
    nf, nd = len(DIR_FRAMES), len(DIR_D)
    vv = f.create_dataset('views', (len(idx), nf, 3, nd, 16, 64, 3), dtype='uint8',
                          chunks=(1, nf, 3, nd, 16, 64, 3), compression='gzip', compression_opts=4)
    vok = f.create_dataset('valid', (len(idx), 3, nd), dtype='bool')
    nbad = 0
    for i, n in enumerate(idx):
        env.set_layout(L.CODE_LAYOUT[int(lay[n])])
        p2 = opos[n, 0].astype(np.float64)
        buf = np.zeros((nf, 3, nd, 16, 64, 3), np.uint8)
        ok = np.zeros((3, nd), bool)
        for ci, c in enumerate(CLASSES):
            b = int(rg[c][n])
            if b < 0:
                continue
            u = BIN_UNIT[b]
            ok[ci] = [not np.any(np.abs(p2 + d * u) > 10) for d in DIR_D]
            for fi, fr in enumerate(DIR_FRAMES):
                for di, d in enumerate(DIR_D):
                    if ok[ci, di]:
                        buf[fi, ci, di] = render(env, pos[n, fr], p2 + d * u)
                if ok[ci, 5]:   # d = 0 is the real frame (A-2 static)
                    nbad += int(not np.array_equal(buf[fi, ci, 5], g['self_vision'][n, fr]))
        vv[i], vok[i] = buf, ok
        if (i + 1) % 25 == 0:
            print(f'{i + 1}/{len(idx)}', flush=True)
    return nbad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('r20', 'dir'), required=True)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--n_parts', type=int, default=1)
    ap.add_argument('--n_ep', type=int, default=600)
    args = ap.parse_args()
    cfg = load_config(os.path.join(_PI, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml'))
    env = creator.create_environment(cfg.environment)
    env.init()
    env.off_display()
    idx = np.array_split(np.arange(args.n_ep), args.n_parts)[args.part]
    od = os.path.join(OUT, f'views_{args.mode}')
    os.makedirs(od, exist_ok=True)
    with h5py.File(os.path.join(od, f'views_part{args.part}of{args.n_parts}.h5'), 'w') as f:
        f.create_dataset('idx', data=idx)
        nbad = (run_r20 if args.mode == 'r20' else run_dir)(env, idx, f)
        f.attrs['rerender_mismatch'] = nbad
    print('rerender mismatches', nbad, 'saved part', args.part, flush=True)


if __name__ == '__main__':
    main()
