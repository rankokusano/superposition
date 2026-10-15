"""
v10 (user request 2026-10-16, item 3; EXPLORATORY, not pre-registered):
is B1 A-1 seed3 a critic whose state value (overall Q level) followed the
colour while its best direction stayed with the place?

For the v3 A-1 critic, B1 A-1 seed0 and seed3 (21x21 grid, other agent at
(0,0), the B1 arrays), in each layout:
  - V(s) = mean of Q over the 8 probe actions ("overall level"),
    A(s, a) = Q(s, a) - V(s) ("directional part");
    corr of V between layouts, corr of A between layouts (same s, same a);
    corr of max-Q map and of V with the true reward map of the same layout
    (red - green, A-1's camera, same placement);
  - best direction (K=64): mean angle difference to red in that layout and
    to red's original place (-9, 9), and to (-9, -9); points within 2 of
    either reference excluded (as Sec.5.0).
Figure: reward map / Q-value map / best-direction map, both layouts, per critic.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR, git_hash  # noqa
sys.path.insert(0, PI_DIR)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa

GRID = np.arange(-10, 11, dtype=float)
X, Y = np.meshgrid(GRID, GRID, indexing='ij')
TL, BL = np.array([-9.0, 9.0]), np.array([-9.0, -9.0])
CRITICS = [('v3 A-1 critic', 'v3_reference_a1'), ('B1 A-1 seed0', 'a1_seed0'), ('B1 A-1 seed3', 'a1_seed3')]
COL = {'Red': '#d62728', 'Green': '#2ca02c', 'Blue': '#1f4fd6', 'Cyan': '#17becf'}


def reward_maps():
    env = creator.create_environment(load_config(os.path.join(
        PI_DIR, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml')).environment)
    env.init()
    env.off_display()
    env.reset()
    out = {}
    for lo in layout.LAYOUT_NAMES:
        env.set_layout(lo)
        m = np.zeros((21, 21))
        for i, x in enumerate(GRID):
            for j, y in enumerate(GRID):
                env.self_agent.p, env.other_agent.p = np.array([x, y]), np.array([0.0, 0.0])
                env.world.set_camera('self')
                env.world.draw(env.self_agent, env.other_agent)
                f = layout.color_fractions(env.world.capture())
                m[i, j] = f['Red'] - f['Green']
        out[lo] = m
    return out


def corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def angdiff(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def angle_to(best, p):
    keep = (np.hypot(X - TL[0], Y - TL[1]) > 2) & (np.hypot(X - BL[0], Y - BL[1]) > 2)
    return float(angdiff(best, np.degrees(np.arctan2(p[1] - Y, p[0] - X)))[keep].mean())


def main():
    out_dir = os.path.join(OUT_ROOT, 'b1', 'diag_a1_seed3')
    os.makedirs(out_dir, exist_ok=True)
    rm = reward_maps()
    res = {'git_commit': git_hash(), 'note': 'exploratory, not pre-registered',
           'corr_reward_original_vs_swap': corr(rm['original'], rm['swap_rg']), 'critics': {}}
    arrs = {}
    for name, lab in CRITICS:
        d = np.load(os.path.join(OUT_ROOT, 'b1', lab, 'b1_arrays.npz'))
        arrs[lab] = d
        q = {lo: d[f'{lo}_q8'] for lo in layout.LAYOUT_NAMES}
        V = {lo: q[lo].mean(axis=2) for lo in q}
        A = {lo: q[lo] - V[lo][..., None] for lo in q}
        r = {'corr_V_between_layouts': corr(V['original'], V['swap_rg']),
             'corr_A_between_layouts': corr(A['original'], A['swap_rg']),
             'corr_Q_between_layouts': corr(q['original'], q['swap_rg']),
             'V_sd_over_grid': [float(V[lo].std()) for lo in q],
             'A_sd_over_grid': [float(A[lo].std()) for lo in q], 'layouts': {}}
        for lo in layout.LAYOUT_NAMES:
            best = d[f'{lo}_best64']
            red_here = layout.LANDMARKS[lo]['Red']
            r['layouts'][lo] = {
                'corr_qmax_with_reward_map_same_layout': corr(d[f'{lo}_qmax8'], rm[lo]),
                'corr_V_with_reward_map_same_layout': corr(V[lo], rm[lo]),
                'best_dir_mean_angle_to_red_here': angle_to(best, red_here),
                'best_dir_mean_angle_to_red_original_place(-9,9)': angle_to(best, TL),
                'best_dir_mean_angle_to_(-9,-9)': angle_to(best, BL),
            }
        res['critics'][name] = r
    json.dump(res, open(os.path.join(out_dir, 'diag_a1_seed3.json'), 'w'), indent=2)
    np.savez(os.path.join(out_dir, 'reward_maps_a1_other_at_centre.npz'), grid=GRID, **rm)
    print(json.dumps(res, indent=1))

    fig, axes = plt.subplots(len(CRITICS) + 1, 4, figsize=(19, 4.3 * (len(CRITICS) + 1)), constrained_layout=True)
    ext = [-10.5, 10.5, -10.5, 10.5]
    lim = np.abs(rm['original']).max()
    for c, lo in enumerate(layout.LAYOUT_NAMES):
        ax = axes[0, c]
        im = ax.imshow(rm[lo].T, origin='lower', extent=ext, cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_title(f'true reward (red - green)\n{lo}', fontsize=10)
        lm(ax, lo)
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    axes[0, 2].text(0, 0.5, f"corr(reward original, reward swapped) = {res['corr_reward_original_vs_swap']:+.3f}\n"
                    "(a value that followed the colour completely would reach this)", fontsize=10)
    for r_i, (name, lab) in enumerate(CRITICS, start=1):
        d = arrs[lab]
        lo_, hi_ = min(d[f'{lo}_qmax8'].min() for lo in layout.LAYOUT_NAMES), max(d[f'{lo}_qmax8'].max() for lo in layout.LAYOUT_NAMES)
        rr = res['critics'][name]
        for c, lo in enumerate(layout.LAYOUT_NAMES):
            ax = axes[r_i, c]
            im = ax.imshow(d[f'{lo}_qmax8'].T, origin='lower', extent=ext, cmap='hot', vmin=lo_, vmax=hi_)
            ax.set_title(f'{name}: Q value, {lo}\ncorr with reward map {rr["layouts"][lo]["corr_qmax_with_reward_map_same_layout"]:+.2f}', fontsize=9)
            lm(ax, lo)
            ax = axes[r_i, 2 + c]
            best = d[f'{lo}_best64']
            dd = angdiff(best, np.degrees(np.arctan2(layout.LANDMARKS[lo]['Red'][1] - Y, layout.LANDMARKS[lo]['Red'][0] - X)))
            th = np.radians(best)
            q = ax.quiver(X, Y, np.cos(th), np.sin(th), dd, cmap='viridis', clim=(0, 180), scale=28, width=0.004)
            L = rr['layouts'][lo]
            ax.set_title(f'{name}: best direction, {lo}\nmean angle to red here {L["best_dir_mean_angle_to_red_here"]:.0f}, '
                         f'to (-9,9) {L["best_dir_mean_angle_to_red_original_place(-9,9)"]:.0f}, '
                         f'to (-9,-9) {L["best_dir_mean_angle_to_(-9,-9)"]:.0f} deg', fontsize=8)
            lm(ax, lo)
        fig.colorbar(im, ax=axes[r_i, 1], shrink=0.8)
        fig.colorbar(q, ax=axes[r_i, 3], shrink=0.8, label='angle to red (deg)')
        axes[r_i, 0].set_ylabel(f'{name}\ncorr V {rr["corr_V_between_layouts"]:+.2f}, corr A {rr["corr_A_between_layouts"]:+.2f}', fontsize=9)
    fig.suptitle('A-1 critics: overall value level (V) vs best direction under the red/green swap (other agent at the centre)')
    fig.savefig(os.path.join(out_dir, 'diag_a1_seed3.png'), dpi=110)


def lm(ax, lo):
    for nm, p in layout.LANDMARKS[lo].items():
        ax.scatter(p[0], p[1], marker='s', s=50, c=COL[nm], edgecolors='k', linewidths=0.6, zorder=5)
    ax.set_xlim(-10.8, 10.8)
    ax.set_ylim(-10.8, 10.8)
    ax.set_aspect('equal')


if __name__ == '__main__':
    main()
