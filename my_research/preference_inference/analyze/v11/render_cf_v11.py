"""
v11 M1(d) counterfactual views (docs/v11_experiment_log.md Sec.7.9), CPU/xvfb.
For every frame t = 0..99 of the common test (v11_s1_random test, 300 ep)
and colour X, A-1 is placed at q = p_t + unit(ref_X - p_t) (speed 1),
A-2 at its real (static) position, in the episode's layout, and A-1's own
view is rendered. (t, X) is invalid if q leaves the arena or
|ref_X - p_t| < 2. Model independent; written once to
data/result/v11/cf/cf_views.h5. Check: re-rendering the real p_{t+1} must
reproduce the stored v_{t+1} (all frames of the first 5 episodes).
"""

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
SRC = os.path.join(_PI_DIR, 'data', 'data', 'v11_s1_random', 'data.h5')
OUT = os.path.join(_PI_DIR, 'data', 'result', 'v11', 'cf', 'cf_views.h5')


def render(env, sp, op):
    env.self_agent.p = np.asarray(sp, dtype=np.float64)
    env.other_agent.p = np.asarray(op, dtype=np.float64)
    env.world.set_camera('self')
    env.world.draw(env.self_agent, env.other_agent)
    return np.round(env.world.capture() * 255).astype(np.uint8)


def main():
    cfg = load_config(os.path.join(_PI_DIR, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml'))
    env = creator.create_environment(cfg.environment)
    env.init()
    env.off_display()
    g = h5py.File(SRC, 'r')['test']
    pos, opos, lay, sv = g['self_position'][()], g['other_position'][()], g['layout'][()], g['self_vision']
    N = pos.shape[0]

    # check: re-render real next positions
    mism = 0
    for n in range(5):
        env.set_layout(L.CODE_LAYOUT[int(lay[n])])
        v = sv[n]
        for t in range(100):
            mism += int(not np.array_equal(render(env, pos[n, t + 1], opos[n, t + 1]), v[t + 1]))
    print('re-render check: mismatched frames', mism, 'of 500', flush=True)
    assert mism == 0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with h5py.File(OUT, 'w') as f:
        dv = f.create_dataset('cf_vision', (N, 100, 4, 16, 64, 3), dtype='uint8')
        dval = f.create_dataset('valid', (N, 100, 4), dtype='bool')
        f.attrs['source'] = SRC
        f.attrs['colors'] = ','.join(COLORS)
        for n in range(N):
            name = L.CODE_LAYOUT[int(lay[n])]
            env.set_layout(name)
            buf = np.zeros((100, 4, 16, 64, 3), np.uint8)
            val = np.zeros((100, 4), bool)
            for t in range(100):
                p = pos[n, t].astype(np.float64)
                for k, X in enumerate(COLORS):
                    r = L.LANDMARKS[name][X] - p
                    d = np.linalg.norm(r)
                    q = p + r / max(d, 1e-9)
                    if d < 2 or np.any(np.abs(q) > 10):
                        continue
                    buf[t, k] = render(env, q, opos[n, t])
                    val[t, k] = True
            dv[n] = buf
            dval[n] = val
            if (n + 1) % 50 == 0:
                print(f'{n + 1}/{N}', flush=True)
    print('saved', OUT)


if __name__ == '__main__':
    main()
