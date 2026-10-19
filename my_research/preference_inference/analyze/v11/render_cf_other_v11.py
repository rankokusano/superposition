"""
v11 M1(d)-other counterfactual views (docs/v11_experiment_log.md Sec.7.52), CPU/xvfb.
For every frame t = 0..99 of v11_s2t_green / v11_s2t_red (test, 300 ep) and colour X, A-2 is
placed at q2 = p2_t + unit(ref_X - p2_t) (A-2's step is 1), A-1 at its real next position
p1_{t+1}, in the episode's layout, and A-1's view is rendered. (t, X) is invalid if q2 leaves the
arena or |ref_X - p2_t| < 2. Model independent. Episodes are split into parts so several CPU
processes can run; each part writes data/result/v11/cf_other/<set>_part<i>.h5.
Check: re-rendering the real (p1_{t+1}, p2_{t+1}) must reproduce v_{t+1} (first 5 episodes).
"""

import argparse
import os
import sys

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

from sim_v10 import creator, layout as L  # noqa
from sim_v10.util import load_config  # noqa

COLORS = ('Red', 'Green', 'Blue', 'Cyan')
OUT = os.path.join(_PI_DIR, 'data', 'result', 'v11', 'cf_other')


def render(env, sp, op):
    env.self_agent.p = np.asarray(sp, dtype=np.float64)
    env.other_agent.p = np.asarray(op, dtype=np.float64)
    env.world.set_camera('self')
    env.world.draw(env.self_agent, env.other_agent)
    return np.round(env.world.capture() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, choices=['v11_s2t_green', 'v11_s2t_red'])
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(os.path.join(_PI_DIR, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml'))
    env = creator.create_environment(cfg.environment)
    env.init()
    env.off_display()
    g = h5py.File(os.path.join(_PI_DIR, 'data', 'data', args.set, 'data.h5'), 'r')['test']
    pos, opos, lay, sv = g['self_position'][()], g['other_position'][()], g['layout'][()], g['self_vision']
    N = pos.shape[0]
    eps = np.array_split(np.arange(N), args.nparts)[args.part]

    if args.part == 0:
        mism = 0
        for n in range(5):
            env.set_layout(L.CODE_LAYOUT[int(lay[n])])
            v = sv[n]
            for t in range(100):
                mism += int(not np.array_equal(render(env, pos[n, t + 1], opos[n, t + 1]), v[t + 1]))
        print('re-render check: mismatched frames', mism, 'of 500', flush=True)
        assert mism == 0

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{args.set}_part{args.part}.h5')
    with h5py.File(path + '.tmp', 'w') as f:
        f.create_dataset('episodes', data=eps)
        dv = f.create_dataset('cf_vision', (len(eps), 100, 4, 16, 64, 3), dtype='uint8')
        dval = f.create_dataset('valid', (len(eps), 100, 4), dtype='bool')
        f.attrs['colors'] = ','.join(COLORS)
        for i, n in enumerate(eps):
            name = L.CODE_LAYOUT[int(lay[n])]
            env.set_layout(name)
            buf = np.zeros((100, 4, 16, 64, 3), np.uint8)
            val = np.zeros((100, 4), bool)
            for t in range(100):
                p2 = opos[n, t].astype(np.float64)
                for k, X in enumerate(COLORS):
                    r = L.LANDMARKS[name][X] - p2
                    d = np.linalg.norm(r)
                    q = p2 + r / max(d, 1e-9)
                    if d < 2 or np.any(np.abs(q) > 10):
                        continue
                    buf[t, k] = render(env, pos[n, t + 1], q)
                    val[t, k] = True
            dv[i] = buf
            dval[i] = val
            if (i + 1) % 10 == 0:
                print(f'{i + 1}/{len(eps)}', flush=True)
    os.replace(path + '.tmp', path)
    print('saved', path)


if __name__ == '__main__':
    main()
