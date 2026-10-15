"""
Thesis figure (v10 stage A): A-2's actor (v3_rl_a2_actor.pth, trained in
the original layout to like Green) in the original vs the red/green-swapped
layout. Episodes are paired (same seeds -> same A-2 start position).

(a) original layout, (b) swapped layout: N_TRAJ sample trajectories (thin),
    start (open circle) and t=100 position (filled) of all 3000 train episodes.
(c) mean +- sd distance of A-2 to the current Green and to Green's original
    place (-9,-9), per timestep, swapped layout; original layout for reference.
Outputs PNG (300 dpi) and PDF to data/result/v10_swap/stageA/a2_terminal/.
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
from matplotlib.patches import Rectangle  # noqa

OUT = os.path.join(OUT_ROOT, 'stageA', 'a2_terminal')
N_TRAJ = 40
COL = {'Red': '#d62728', 'Green': '#2ca02c', 'Blue': '#1f4fd6', 'Cyan': '#17becf'}
LM = {
    'original': {'Red': (-9, 9), 'Green': (-9, -9), 'Blue': (9, -9), 'Cyan': (9, 9)},
    'swap_rg': {'Red': (-9, -9), 'Green': (-9, 9), 'Blue': (9, -9), 'Cyan': (9, 9)},
}
GREEN_OLD = np.array([-9.0, -9.0])
GREEN_NEW = np.array([-9.0, 9.0])


def load(ds):
    with h5py.File(os.path.join(PI_DIR, 'data', 'data', ds, 'data.h5'), 'r') as f:
        return f['train/other_position'][()]


def arena(ax, lo, title):
    ax.add_patch(Rectangle((-10, -10), 20, 20, fill=False, ec='0.3', lw=1))
    for nm, (x, y) in LM[lo].items():
        # landmark cubes sit outside the arena (centre +-13); drawn at the corner
        ax.add_patch(Rectangle((x + np.sign(x) * 1.0 - 1.0, y + np.sign(y) * 1.0 - 1.0), 2, 2,
                               color=COL[nm], zorder=4))
        ax.text(x + np.sign(x) * 1.0, y - np.sign(y) * 1.9, nm, ha='center', va='center',
                fontsize=8, color='0.15', zorder=5)
    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_aspect('equal')
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    ax.set_title(title, fontsize=10)


def main():
    p_o, p_s = load('r2_a1random_a2rl'), load('v10_swap_r2')
    fig = plt.figure(figsize=(12.5, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
    for k, (lo, p, ttl) in enumerate([('original', p_o, '(a) Original layout (as in training)'),
                                       ('swap_rg', p_s, '(b) Red and green swapped')]):
        ax = fig.add_subplot(gs[0, k])
        arena(ax, lo, ttl)
        for i in range(N_TRAJ):
            ax.plot(p[i, :, 0], p[i, :, 1], color='0.45', lw=0.6, alpha=0.6, zorder=2)
        ax.scatter(p[:N_TRAJ, 0, 0], p[:N_TRAJ, 0, 1], s=14, facecolors='white', edgecolors='0.35', lw=0.7, zorder=3,
                   label=f'start ({N_TRAJ} sample episodes)')
        ax.scatter(p[:, -1, 0], p[:, -1, 1], s=6, c='k', alpha=0.5, zorder=3, label='end, t=100 (all 3000)')
        if k == 0:
            ax.legend(fontsize=7, loc='center', framealpha=0.9)
        d_new = np.linalg.norm(p[:, -1] - (GREEN_OLD if lo == 'original' else GREEN_NEW), axis=1)
        ax.text(0, -12.6, f'end within 5 of green: {(d_new <= 5).mean() * 100:.0f}%'
                + ('' if lo == 'original' else
                   f'\nend within 5 of green\'s old place: {(np.linalg.norm(p[:, -1] - GREEN_OLD, axis=1) <= 5).mean() * 100:.0f}%'),
                ha='center', va='top', fontsize=8)
    ax = fig.add_subplot(gs[0, 2])
    t = np.arange(p_o.shape[1])
    for lab, d, c, ls in [
        ('swapped: to green (-9, 9)', np.linalg.norm(p_s - GREEN_NEW, axis=2), COL['Green'], '-'),
        ("swapped: to green's old place (-9, -9), now red", np.linalg.norm(p_s - GREEN_OLD, axis=2), COL['Red'], '-'),
        ('original: to green (-9, -9)', np.linalg.norm(p_o - GREEN_OLD, axis=2), '0.35', '--'),
    ]:
        m, s = d.mean(0), d.std(0)
        ax.plot(t, m, color=c, ls=ls, lw=2, label=lab)
        ax.fill_between(t, m - s, m + s, color=c, alpha=0.12, lw=0)
    ax.set_xlabel('time step')
    ax.set_ylabel('distance of A-2 (mean ± sd, 3000 episodes)')
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 24)
    ax.grid(color='0.92')
    ax.legend(fontsize=8, loc='center right')
    ax.set_title('(c) A-2 goes to where green was, not to green', fontsize=10)
    fig.suptitle("A-2's actor (trained to approach green in the original layout) after red and green are swapped",
                 fontsize=11)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_a2_actor_swap.{ext}'), dpi=300)
    print('saved')


if __name__ == '__main__':
    main()
