"""
v10 copy of the root simulation/collect_data.py grid mode (config
collect/grid.yml: num_div 21, both agents on the 21x21 grid, other_vision
and *_no_agent views) rendered with sim_v10 in a chosen layout. Vision is
stored as uint8 (the root grid is float32; the render is 8-bit quantized,
so round(v*255) is lossless and exp/loader.py / util.scale_vision accept
uint8). Used for Fig.5 (analyze_vpt) in the swapped layout.

--n_other limits the number of other-agent positions (outer loop), for the
bit-identity check against the root grid.

Usage (inside Docker, from /work/my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_grid_v10.py --layout swap_rg \
        --save_path data/data/v10_swap_grid/data.h5
"""

import argparse
import os
import sys

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PI_DIR)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa

GRID_CONFIG = os.path.join(_PI_DIR, 'sim_v10', 'config', 'collect', 'grid.yml')


def to_uint8(v):
    return np.round(v * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', required=True, choices=list(layout.LAYOUT_NAMES))
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--n_other', type=int, default=None)
    args = parser.parse_args()

    config = load_config(GRID_CONFIG)
    env = creator.create_environment(config.environment)
    env.init()
    env.set_camera('self')
    env.off_display()
    env.set_layout(args.layout)

    wc = env.world.config
    nd = config.num_div
    N = nd * nd
    n_other = N if args.n_other is None else args.n_other
    v_shape = (n_other, N, wc.camera.agent.height, wc.camera.agent.width, 3)
    p_shape = (n_other, N, 2)

    save_path = os.path.join(_PI_DIR, args.save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, 'w') as h5:
        h5.attrs['layout'] = args.layout
        h5.attrs['source'] = 'collect_grid_v10.py (copy of root collect_data.py grid mode)'
        d = {k: h5.create_dataset('train/' + k, v_shape, dtype='uint8')
             for k in ['self_vision', 'other_vision', 'self_vision_no_agent', 'other_vision_no_agent']}
        d['self_position'] = h5.create_dataset('train/self_position', p_shape, dtype='f')
        d['other_position'] = h5.create_dataset('train/other_position', p_shape, dtype='f')

        n = 0
        for ox in np.linspace(wc.xmin, wc.xmax, nd):
            for oy in np.linspace(wc.ymin, wc.ymax, nd):
                if n >= n_other:
                    break
                op = np.array([ox, oy])
                buf = {k: np.zeros(v_shape[1:], np.uint8) for k in
                       ['self_vision', 'other_vision', 'self_vision_no_agent', 'other_vision_no_agent']}
                sps = np.zeros((N, 2), np.float32)
                t = 0
                for sx in np.linspace(wc.xmin, wc.xmax, nd):
                    for sy in np.linspace(wc.ymin, wc.ymax, nd):
                        sp = np.array([sx, sy])
                        env.set_agent_pos(sp, op)
                        env.set_camera('self')
                        buf['self_vision'][t] = to_uint8(env.capture())
                        env.set_camera('other')
                        buf['other_vision'][t] = to_uint8(env.capture())
                        env.set_camera('self')
                        env.world.set_visible(False, False)
                        buf['self_vision_no_agent'][t] = to_uint8(env.capture())
                        env.set_camera('other')
                        env.world.set_visible(False, False)
                        buf['other_vision_no_agent'][t] = to_uint8(env.capture())
                        sps[t] = sp
                        t += 1
                for k, v in buf.items():
                    d[k][n] = v
                d['self_position'][n] = sps
                d['other_position'][n] = np.tile(op, (N, 1))
                n += 1
                if n % 21 == 0:
                    print(f'{n}/{n_other} other positions done', flush=True)
    print('Saved:', save_path)


if __name__ == '__main__':
    main()
