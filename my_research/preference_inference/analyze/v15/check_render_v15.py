"""
v15 rendering checks (docs/v15_experiment_log.md Sec.4.1):
  --mode ref   render sim_v10 (original Environment) on a position grid, both layouts
  --mode v15   render World15 with no gray at the same positions; and every gray state
  --mode compare  bit-compare ref vs v15; gray pixels hit no colour mask; gray vs floor values
Each render mode runs in its own process (one GL context per process).
"""

import argparse
import os
import sys

import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
OUT = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'render_check')
GRID = np.linspace(-10, 10, 9)


def positions():
    ps = [(x, y) for x in GRID for y in GRID]
    rng = np.random.RandomState(0)
    others = [rng.uniform(-10, 10, 2) for _ in ps]
    return [np.array(p, float) for p in ps], others


def render(env, set_state, states):
    ps, os_ = positions()
    out = {}
    for name, st in states.items():
        set_state(st)
        vs = []
        for p, o in zip(ps, os_):
            env.self_agent.p, env.other_agent.p = p.copy(), o.copy()
            for cam in ('self', 'other'):
                env.world.set_camera(cam)
                env.world.draw(env.self_agent, env.other_agent)
                vs.append(np.round(env.world.capture() * 255).astype(np.uint8))
        out[name] = np.stack(vs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['ref', 'v15', 'compare'])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from sim_v10 import layout
    if args.mode == 'ref':
        from sim_v10 import creator
        from sim_v10.util import load_config
        cfg = load_config(os.path.join(_PI_DIR, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml'))
        env = creator.create_environment(cfg.environment)
        env.init(); env.off_display(); env.reset()
        out = render(env, env.set_layout, {l: l for l in layout.LAYOUT_NAMES})
        np.savez_compressed(os.path.join(OUT, 'ref.npz'), **out)
    elif args.mode == 'v15':
        from v15core.env import Env15
        from v15core.world import COLORS
        env = Env15('random', 'stay')
        env.init(); env.reset()
        states = {l: (l, None) for l in layout.LAYOUT_NAMES}
        for l in layout.LAYOUT_NAMES:
            for g in COLORS:
                states[f'{l}_gray{g}'] = (l, g)
            states[f'{l}_allgray'] = (l, 'all')

        def set_state(st):
            if st[1] == 'all':
                env.world.set_episode(st[0], None)
                for c in COLORS:
                    env.world.eat(c)
            else:
                env.world.set_episode(*st)
        out = render(env, set_state, states)
        np.savez_compressed(os.path.join(OUT, 'v15.npz'), **out)
    else:
        from v11core.pleasure import color_mask
        from v15core.world import COLORS
        ref, v15 = np.load(os.path.join(OUT, 'ref.npz')), np.load(os.path.join(OUT, 'v15.npz'))
        for l in layout.LAYOUT_NAMES:
            same = np.array_equal(ref[l], v15[l])
            print(f'{l}: v15 (no gray) bit-identical to sim_v10: {same}  '
                  f'(max |diff| {np.abs(ref[l].astype(int) - v15[l]).max()}, n views {len(ref[l])})')
        for l in layout.LAYOUT_NAMES:
            base, allg = v15[l], v15[f'{l}_allgray']
            changed = np.any(base != allg, -1)
            gp = allg[changed]
            hits = {c: int(color_mask(gp, c).sum()) for c in COLORS}
            print(f'{l}: pixels that change when all cubes are gray: {changed.sum()}; '
                  f'colour-mask hits among gray pixels {hits}; gray values min {gp.min(0)} max {gp.max(0)}')
            for g in COLORS:
                v = v15[f'{l}_gray{g}']
                n_g = sum(int(color_mask(v, g).sum()) for _ in [0])
                others = {c: int(color_mask(v, c).sum()) == int(color_mask(base, c).sum()) for c in COLORS if c != g}
                print(f'  gray {g}: {g} pixels left {n_g}; other colours unchanged {others}')
        # floor / ground values for comparison (self view at centre, lower rows)
        print('floor/ground pixel values (unique rows of bottom line, original, first view):',
              np.unique(v15['original'][0][-1].reshape(-1, 3), axis=0)[:6])


if __name__ == '__main__':
    main()
