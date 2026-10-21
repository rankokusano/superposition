"""
v12 proposal C data (docs/v12_experiment_log.md Sec.43/45): copy a stage-2 dataset and add
  self_vision_in : A-1's view used as the network INPUT; equal to self_vision on visible steps and re-rendered with
                   A-2 moved outside the arena (1e4, 1e4) on hidden steps (A-2 vanishes, nothing else changes);
  a2_hidden      : (N, T) uint8, 1 on hidden steps.
self_vision (the prediction target) is unchanged. Schedule per episode (fixed seed, same for every model): t = 0
visible; the first visible interval has length U{1..L_max}, then hidden / visible intervals alternate with lengths
U{L_min..L_max} (default 10..30, about 50% hidden).
Render in parts (each under its own xvfb-run), then merge:
  python analyze/v12/make_hide_a2_v12.py --src v11_s2_green --split train --part 0 --nparts 6
  python analyze/v12/make_hide_a2_v12.py --src v11_s2_green --merge --nparts 6
Writes data/data/v12_hide_<src>/data.h5 (parts in data/data/v12_hide_<src>/parts/).
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(_PI, 'data', 'data')


def schedule(n_ep, T, seed, lmin, lmax):
    rng = np.random.RandomState(seed)
    H = np.zeros((n_ep, T), np.uint8)
    for e in range(n_ep):
        t, hid = int(rng.randint(1, lmax + 1)), True
        while t < T:
            L = int(rng.randint(lmin, lmax + 1))
            if hid:
                H[e, t:t + L] = 1
            t += L
            hid = not hid
    return H


def out_dir(src):
    return os.path.join(DATA, f'v12_hide_{src}')


def render_part(a):
    os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
    sys.path.insert(0, _PI)
    from sim_v10 import creator, layout as L
    from sim_v10.util import load_config
    cfg = load_config(os.path.join(_PI, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml'))
    env = creator.create_environment(cfg.environment); env.init(); env.off_display()

    def render(s, o):
        env.self_agent.p = np.asarray(s, float); env.other_agent.p = np.asarray(o, float)
        env.world.set_camera('self'); env.world.draw(env.self_agent, env.other_agent)
        return np.round(env.world.capture() * 255).astype(np.uint8)

    g = h5py.File(os.path.join(DATA, a.src, 'data.h5'), 'r')[a.split]
    N, T = g['self_position'].shape[:2]
    H = schedule(N, T, a.seed + (0 if a.split == 'train' else 1), a.lmin, a.lmax)
    eps = np.array_split(np.arange(N), a.nparts)[a.part]
    sp, op = g['self_position'][eps[0]:eps[-1] + 1], g['other_position'][eps[0]:eps[-1] + 1]
    lay_t = g['layout_t'][eps[0]:eps[-1] + 1]
    sv = g['self_vision'][eps[0]:eps[-1] + 1]
    out = sv.copy()
    n, same, t0 = 0, 0, time.time()
    for i, e in enumerate(eps):
        for t in range(T):
            env.set_layout(L.CODE_LAYOUT[int(lay_t[i, t])])
            if t % 25 == 0:     # check: re-render WITH A-2 equals the stored view
                same += int(np.array_equal(render(sp[i, t], op[i, t]), sv[i, t])); n += 1
            if H[e, t]:
                out[i, t] = render(sp[i, t], (1e4, 1e4))
    os.makedirs(os.path.join(out_dir(a.src), 'parts'), exist_ok=True)
    np.savez(os.path.join(out_dir(a.src), 'parts', f'{a.split}_{a.part}.npz'), eps=eps, sv_in=out)
    print(f'{a.src} {a.split} part {a.part}: {len(eps)} episodes, re-render check {same}/{n}, '
          f'{time.time() - t0:.0f} s', flush=True)


def merge(a):
    src = h5py.File(os.path.join(DATA, a.src, 'data.h5'), 'r')
    dst_path = os.path.join(out_dir(a.src), 'data.h5')
    with h5py.File(dst_path, 'w') as dst:
        dst.attrs.update({'source': a.src, 'seed': a.seed, 'lmin': a.lmin, 'lmax': a.lmax})
        for split in src:
            g, o = src[split], dst.create_group(split)
            for k in g:
                o.create_dataset(k, data=g[k][()])
            N, T = g['self_position'].shape[:2]
            H = schedule(N, T, a.seed + (0 if split == 'train' else 1), a.lmin, a.lmax)
            sv_in = np.empty_like(g['self_vision'][()])
            for p in range(a.nparts):
                z = np.load(os.path.join(out_dir(a.src), 'parts', f'{split}_{p}.npz'))
                sv_in[z['eps']] = z['sv_in']
            o.create_dataset('self_vision_in', data=sv_in)
            o.create_dataset('a2_hidden', data=H)
            # checks: visible steps untouched; hidden steps differ only where A-2 was (dark pixels)
            vis = H == 0
            sv = g['self_vision'][()]
            d = np.abs(sv_in.astype(np.int16) - sv).max(-1) > 0
            dark = sv.max(-1) < 26
            print(f'{a.src} {split}: hidden fraction {H.mean():.3f} (t>0 {H[:, 1:].mean():.3f}); '
                  f'visible steps changed {int(d[vis].sum())} px; hidden steps changed outside A-2 pixels '
                  f'{int((d & ~dark)[~vis].sum())} px; hidden frames with A-2 removed {(d.any((2, 3)) & ~vis).sum()}'
                  f' / {(~vis).sum()}', flush=True)
    print('saved', dst_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--nparts', type=int, default=6)
    ap.add_argument('--merge', action='store_true')
    ap.add_argument('--seed', type=int, default=12043)
    ap.add_argument('--lmin', type=int, default=10)
    ap.add_argument('--lmax', type=int, default=30)
    a = ap.parse_args()
    merge(a) if a.merge else render_part(a)


if __name__ == '__main__':
    main()
