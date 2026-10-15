"""
v10 gate 6.0 (docs/v10_experiment_log.md Sec.5.1, checks 2-4).

On the 21x21 grid (x, y in -10..10, as the original paper's `grid`), with
A-1 on the grid point (self camera) and A-2 moved out of the arena (50, 50):
  2. colour fractions in both layouts; max |Red_swap - Green_orig|,
     |Green_swap - Red_orig|, |Blue_swap - Blue_orig|, |Cyan_swap - Cyan_orig|
     (pre-registered expectation: all exactly 0)
  3. figure: Red/Green/Blue/Cyan fractions and A-1 reward (Red - Green) maps,
     both layouts
  4. figure: views at the same position in both layouts, plus the swap view
     at (x, y) next to the original view at the vertically mirrored (x, -y)
     -- they must NOT be identical (blue/cyan do not move); the pixel
     difference is counted on the whole grid.
Also: switching layout and back reproduces the original view bit-identically.

All outputs go to data/result/v10_swap/gate60/.

Usage (inside Docker, from /work/my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python analyze/v10/gate60_layout_check.py
"""

import json
import os
import sys

import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import git_hash  # noqa

OUT_DIR = os.path.join(_PI_DIR, 'data', 'result', 'v10_swap', 'gate60')
ENV_CONFIG = os.path.join(_PI_DIR, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml')
GRID = np.arange(-10, 11, dtype=float)  # 21 points
FAR = np.array([50.0, 50.0])
COLORS = ['Red', 'Green', 'Blue', 'Cyan']
EXAMPLE_POS = [(-6.0, 6.0), (-6.0, -6.0), (0.0, 0.0), (6.0, 3.0)]


def render(env, pos, name):
    env.set_layout(name)
    env.self_agent.p = np.array(pos, dtype=float)
    env.other_agent.p = FAR.copy()
    env.world.set_camera('self')
    env.world.draw(env.self_agent, env.other_agent)
    return env.world.capture().copy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_config(ENV_CONFIG)
    env = creator.create_environment(cfg.environment)
    env.init()
    env.off_display()
    env.reset()

    n = len(GRID)
    frac = {lo: {c: np.zeros((n, n)) for c in COLORS} for lo in layout.LAYOUT_NAMES}
    mirror_diff_px = np.zeros((n, n), dtype=int)   # swap(x,y) vs original(x,-y)
    mirror_rg_err = 0.0                             # red/green fractions of that pair
    toggle_ok = True
    for i, x in enumerate(GRID):
        for j, y in enumerate(GRID):
            v_o = render(env, (x, y), layout.ORIGINAL)
            v_s = render(env, (x, y), layout.SWAP_RG)
            v_o2 = render(env, (x, y), layout.ORIGINAL)
            toggle_ok &= bool(np.array_equal(v_o, v_o2))
            for lo, v in ((layout.ORIGINAL, v_o), (layout.SWAP_RG, v_s)):
                f = layout.color_fractions(v)
                for c in COLORS:
                    frac[lo][c][i, j] = f[c]
            v_m = render(env, (x, -y), layout.ORIGINAL)
            mirror_diff_px[i, j] = int((np.abs(v_s - v_m).max(axis=2) > 0).sum())
            fm, fs = layout.color_fractions(v_m), layout.color_fractions(v_s)
            mirror_rg_err = max(mirror_rg_err, abs(fs['Red'] - fm['Red']), abs(fs['Green'] - fm['Green']))

    o, s = frac[layout.ORIGINAL], frac[layout.SWAP_RG]
    px = 16 * 64
    errs = {
        'red_swap_vs_green_orig': float(np.abs(s['Red'] - o['Green']).max()),
        'green_swap_vs_red_orig': float(np.abs(s['Green'] - o['Red']).max()),
        'blue_swap_vs_blue_orig': float(np.abs(s['Blue'] - o['Blue']).max()),
        'cyan_swap_vs_cyan_orig': float(np.abs(s['Cyan'] - o['Cyan']).max()),
    }
    result = {
        'gate': '6.0 checks 2 and 4 (v10_experiment_log Sec.5.1)',
        'git_commit': git_hash(),
        'grid': 'x,y in -10..10 step 1 (21x21), A-2 at (50,50), self camera',
        'max_abs_fraction_error': errs,
        'max_abs_pixel_count_error': {k: round(v * px) for k, v in errs.items()},
        'check2_pass': all(v == 0.0 for v in errs.values()),
        'layout_toggle_back_bit_identical': toggle_ok,
        'mirror_check': {
            'what': 'swap view at (x,y) vs original view at (x,-y)',
            'n_grid_points_identical': int((mirror_diff_px == 0).sum()),
            'min_differing_pixels': int(mirror_diff_px.min()),
            'mean_differing_pixels': float(mirror_diff_px.mean()),
            'max_red_green_fraction_error': float(mirror_rg_err),
        },
        'reward_A1_range': {lo: [float((frac[lo]['Red'] - frac[lo]['Green']).min()),
                                 float((frac[lo]['Red'] - frac[lo]['Green']).max())]
                            for lo in layout.LAYOUT_NAMES},
    }
    with open(os.path.join(OUT_DIR, 'gate60_layout_check.json'), 'w') as f:
        json.dump(result, f, indent=2)
    np.savez(os.path.join(OUT_DIR, 'gate60_fractions.npz'), grid=GRID,
             mirror_diff_px=mirror_diff_px,
             **{f'{lo}_{c}': frac[lo][c] for lo in layout.LAYOUT_NAMES for c in COLORS})
    print(json.dumps(result, indent=2))

    plot_fraction_maps(frac)
    plot_views(env)


def plot_fraction_maps(frac):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    vmax = max(frac[lo][c].max() for lo in layout.LAYOUT_NAMES for c in COLORS)
    rmax = max(np.abs(frac[lo]['Red'] - frac[lo]['Green']).max() for lo in layout.LAYOUT_NAMES)
    ext = [GRID[0] - .5, GRID[-1] + .5, GRID[0] - .5, GRID[-1] + .5]
    for r, lo in enumerate(layout.LAYOUT_NAMES):
        for c_i, c in enumerate(COLORS):
            ax = axes[r, c_i]
            im = ax.imshow(frac[lo][c].T, origin='lower', extent=ext, cmap='Blues', vmin=0, vmax=vmax)
            ax.set_title(f'{lo}: {c} fraction', fontsize=10)
            annotate(ax, lo)
        ax = axes[r, 4]
        im2 = ax.imshow((frac[lo]['Red'] - frac[lo]['Green']).T, origin='lower', extent=ext,
                        cmap='RdBu_r', vmin=-rmax, vmax=rmax)
        ax.set_title(f'{lo}: A-1 reward (Red - Green)', fontsize=10)
        annotate(ax, lo)
    fig.colorbar(im, ax=axes[:, :4], shrink=0.8, label='fraction of 16x64 pixels')
    fig.colorbar(im2, ax=axes[:, 4], shrink=0.8, label='reward')
    fig.suptitle('v10 gate 6.0: colour fractions and A-1 reward on the 21x21 grid (A-2 out of arena)')
    fig.savefig(os.path.join(OUT_DIR, 'gate60_fraction_reward_maps.png'), dpi=120)
    plt.close(fig)


def annotate(ax, lo):
    for name, p in layout.LANDMARKS[lo].items():
        ax.scatter(p[0], p[1], marker='D', s=40, c='none', edgecolors='k', linewidths=1)
        ax.text(p[0] + .4, p[1] + .4, name, fontsize=7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_views(env):
    rows = len(EXAMPLE_POS)
    fig, axes = plt.subplots(rows, 4, figsize=(18, 1.25 * rows + 1.2), constrained_layout=True)
    arrays = {}
    for r, (x, y) in enumerate(EXAMPLE_POS):
        v_o = render(env, (x, y), layout.ORIGINAL)
        v_s = render(env, (x, y), layout.SWAP_RG)
        v_m = render(env, (x, -y), layout.ORIGINAL)
        d = np.abs(v_s - v_m)
        arrays[f'pos{r}_original'] = v_o
        arrays[f'pos{r}_swap'] = v_s
        arrays[f'pos{r}_original_mirror'] = v_m
        panels = [(v_o, f'original ({x:g},{y:g})'), (v_s, f'swap_rg ({x:g},{y:g})'),
                  (v_m, f'original mirrored ({x:g},{-y:g})'),
                  (d / max(d.max(), 1e-9), f'|swap({x:g},{y:g}) - orig({x:g},{-y:g})|  ({int((d.max(axis=2) > 0).sum())} px differ)')]
        for c, (img, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1), aspect='auto', interpolation='nearest')
            ax.set_title(title, fontsize=9)
            ax.set_xticks([0, 16, 32, 48, 63])
            ax.set_yticks([])
    fig.suptitle('v10 gate 6.0: A-1 view (self camera, 16x64, columns fixed to world bearing), A-2 out of arena')
    fig.savefig(os.path.join(OUT_DIR, 'gate60_view_comparison.png'), dpi=120)
    plt.close(fig)
    np.savez(os.path.join(OUT_DIR, 'gate60_view_examples.npz'), positions=np.array(EXAMPLE_POS), **arrays)


if __name__ == '__main__':
    main()
