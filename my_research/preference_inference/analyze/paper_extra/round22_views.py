"""
Paper round 22 Sec.2.4 (docs/paper_extra_log.md Sec.46): reference views for the other-pathway feature read-out
(model independent, CPU/xvfb, as round21_views.py). For the V1 flow (freeze from T0 = 23, K = 10 other steps), the
feature prediction made at step T0+k-1 (k = 1..10) is a prediction of the Encoder-2 output of frame T0+k (as the FP
monitor: predict_feature(os_t) vs other_vision_encoder(x_{t+1})). So A-1 is placed at its real position of frame
T0+k (frames 24..33) and A-2 at p2 + d unit(L - p2), d = -5..5, L in {L_old, L_new}; both layouts (pre-swap, swapped).
  views[n, k-1, target, layout, d index]   uint8 (16, 64, 3);  valid[n, target, d index]  (|q| <= 10)
Re-render check: d = 0 in the data's current layout equals the data frame T0+k (noswap: pre-swap layout, swap:
swapped layout), for every k.
Usage: xvfb-run --auto-servernum python analyze/paper_extra/round22_views.py --part i --n_parts n
-> data/result/paper_extra/Round22/views/views_part<i>of<n>.h5
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
OUT = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round22', 'views')
T0, K = 23, 10
DREF = np.arange(-5, 6)


def render(env, sp, op):
    env.self_agent.p = np.asarray(sp, dtype=np.float64)
    env.other_agent.p = np.asarray(op, dtype=np.float64)
    env.world.set_camera('self')
    env.world.draw(env.self_agent, env.other_agent)
    return np.round(env.world.capture() * 255).astype(np.uint8)


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
    nd = len(DREF)
    with h5py.File(os.path.join(OUT, f'views_part{args.part}of{args.n_parts}.h5'), 'w') as f:
        f.create_dataset('idx', data=idx)
        vv = f.create_dataset('views', (len(idx), K, 2, 2, nd, 16, 64, 3), dtype='uint8',
                              chunks=(1, K, 2, 2, nd, 16, 64, 3), compression='gzip', compression_opts=4)
        vok = f.create_dataset('valid', (len(idx), 2, nd), dtype='bool')
        f.attrs['t0'] = T0
        f.attrs['d'] = DREF
        for i, n in enumerate(idx):
            lay0 = L.CODE_LAYOUT[int(lay[n])]
            lay1 = L.other_layout(lay0)
            p2 = opos[n, 0].astype(np.float64)
            tg = (np.asarray(L.LANDMARKS[lay0]['Red'], np.float64), np.asarray(L.LANDMARKS[lay1]['Red'], np.float64))
            buf = np.zeros((K, 2, 2, nd, 16, 64, 3), np.uint8)
            ok = np.zeros((2, nd), bool)
            for ti, t in enumerate(tg):
                u = (t - p2) / max(np.linalg.norm(t - p2), 1e-9)
                for di, d in enumerate(DREF):
                    ok[ti, di] = not np.any(np.abs(p2 + d * u) > 10)
            for li, ln in enumerate((lay0, lay1)):
                env.set_layout(ln)
                for k in range(K):
                    sp = pos[n, T0 + k + 1].astype(np.float64)
                    for ti, t in enumerate(tg):
                        u = (t - p2) / max(np.linalg.norm(t - p2), 1e-9)
                        for di, d in enumerate(DREF):
                            if ok[ti, di]:
                                buf[k, ti, li, di] = render(env, sp, p2 + d * u)
                    # re-render check (d = 0 is the real frame)
                    ref = (g if li == 0 else gs)['self_vision'][n, T0 + k + 1]
                    nbad += int(not np.array_equal(buf[k, 0, li, int(np.where(DREF == 0)[0][0])], ref))
            vv[i], vok[i] = buf, ok
            if (i + 1) % 25 == 0:
                print(f'{i + 1}/{len(idx)}', flush=True)
        f.attrs['rerender_mismatch'] = nbad
    print('rerender mismatches', nbad, 'saved part', args.part, flush=True)


if __name__ == '__main__':
    main()
