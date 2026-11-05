"""
Paper round 21 (docs/paper_extra_log.md Sec.45): reference views for the self read-out d* (as
analyze/v11/render_cfdist_v11.py, CPU/xvfb). Model independent. From A-1's position p_T0 at the freeze start
T0 (20 = V0 and the check-3 flow, 23 = V1 flow), A-1 is placed d = 0..10 unit steps toward a target, A-2 at its real
(static) position; invalid if outside the arena (|q| > 10). Targets:
  views_lt[n, T0 index, target, layout, d]  target 0 = L_old (Red reference of the pre-swap layout),
                                            target 1 = L_new (Red reference of the swapped layout = Green of the
                                            pre-swap layout); layout 0 = pre-swap layout, 1 = swapped layout;
  views4[n, colour, d]                      T0 = 20, pre-swap layout, colours Red / Green / Blue / Cyan (= the
                                            existing cf_dist_views for experiment 1's g, for check 3).
Positions are read from data_noswap.h5 (A-1's path and A-2's position are the same in both variants, Sec.1.2).
Re-render check: frame T0 of data_noswap (pre-swap layout) and frame 23 of data_swap (swapped layout).
Usage: xvfb-run --auto-servernum python analyze/paper_extra/round21_views.py --part i --n_parts n
-> data/result/paper_extra/Round21/views/views_part<i>of<n>.h5
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

DATA = os.path.join(_PI, 'data', 'data', 'paper_r21_swap')
OUT = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round21', 'views')
COLORS = ('Red', 'Green', 'Blue', 'Cyan')
T0S = (20, 23)
DMAX = 10


def render(env, sp, op):
    env.self_agent.p = np.asarray(sp, dtype=np.float64)
    env.other_agent.p = np.asarray(op, dtype=np.float64)
    env.world.set_camera('self')
    env.world.draw(env.self_agent, env.other_agent)
    return np.round(env.world.capture() * 255).astype(np.uint8)


def toward(env, p, target, op):
    u = target - p
    u = u / max(np.linalg.norm(u), 1e-9)
    buf = np.zeros((DMAX + 1, 16, 64, 3), np.uint8)
    val = np.zeros(DMAX + 1, bool)
    for dd in range(DMAX + 1):
        q = p + dd * u
        if np.any(np.abs(q) > 10):
            continue
        buf[dd] = render(env, q, op)
        val[dd] = True
    return buf, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--n_parts', type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(os.path.join(_PI, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml'))
    env = creator.create_environment(cfg.environment)
    env.init()
    env.off_display()
    g = h5py.File(os.path.join(DATA, 'data_noswap.h5'), 'r')['test']
    gs = h5py.File(os.path.join(DATA, 'data_swap.h5'), 'r')['test']
    pos, opos, lay = g['self_position'][()], g['other_position'][()], g['layout'][()]
    N = pos.shape[0]
    idx = np.array_split(np.arange(N), args.n_parts)[args.part]
    os.makedirs(OUT, exist_ok=True)
    nbad = 0
    with h5py.File(os.path.join(OUT, f'views_part{args.part}of{args.n_parts}.h5'), 'w') as f:
        f.create_dataset('idx', data=idx)
        vlt = f.create_dataset('views_lt', (len(idx), 2, 2, 2, DMAX + 1, 16, 64, 3), dtype='uint8',
                               chunks=(1, 2, 2, 2, DMAX + 1, 16, 64, 3), compression='gzip', compression_opts=4)
        vlt_ok = f.create_dataset('valid_lt', (len(idx), 2, 2, 2, DMAX + 1), dtype='bool')
        v4 = f.create_dataset('views4', (len(idx), 4, DMAX + 1, 16, 64, 3), dtype='uint8',
                              chunks=(1, 4, DMAX + 1, 16, 64, 3), compression='gzip', compression_opts=4)
        v4_ok = f.create_dataset('valid4', (len(idx), 4, DMAX + 1), dtype='bool')
        f.attrs['t0s'] = T0S
        for i, n in enumerate(idx):
            lay0 = L.CODE_LAYOUT[int(lay[n])]
            lay1 = L.other_layout(lay0)
            targets = (L.LANDMARKS[lay0]['Red'], L.LANDMARKS[lay1]['Red'])
            # re-render checks
            env.set_layout(lay0)
            nbad += int(not np.array_equal(render(env, pos[n, 20], opos[n, 20]), g['self_vision'][n, 20]))
            env.set_layout(lay1)
            nbad += int(not np.array_equal(render(env, pos[n, 23], opos[n, 23]), gs['self_vision'][n, 23]))
            buf = np.zeros((2, 2, 2, DMAX + 1, 16, 64, 3), np.uint8)
            ok = np.zeros((2, 2, 2, DMAX + 1), bool)
            for li, ln in enumerate((lay0, lay1)):
                env.set_layout(ln)
                for ti, t0 in enumerate(T0S):
                    p = pos[n, t0].astype(np.float64)
                    for k, tg in enumerate(targets):
                        buf[ti, k, li], ok[ti, k, li] = toward(env, p, np.asarray(tg, np.float64), opos[n, t0])
            vlt[i], vlt_ok[i] = buf, ok
            env.set_layout(lay0)
            b4 = np.zeros((4, DMAX + 1, 16, 64, 3), np.uint8)
            o4 = np.zeros((4, DMAX + 1), bool)
            p = pos[n, 20].astype(np.float64)
            for k, X in enumerate(COLORS):
                b4[k], o4[k] = toward(env, p, np.asarray(L.LANDMARKS[lay0][X], np.float64), opos[n, 20])
            v4[i], v4_ok[i] = b4, o4
            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{len(idx)}', flush=True)
        f.attrs['rerender_mismatch'] = nbad
    print('rerender mismatches', nbad, 'saved part', args.part, flush=True)


if __name__ == '__main__':
    main()
