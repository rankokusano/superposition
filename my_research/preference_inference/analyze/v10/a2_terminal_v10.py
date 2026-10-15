"""
v10 stage A: where does A-2's actor (v3_rl_a2_actor.pth, trained in the
original layout to like Green) go in the swapped layout -- to the new Green
(-9, 9) or to Green's old place (-9, -9)? (docs/v10_experiment_log.md Sec.5.2)

Compares r2_a1random_a2rl (original) and v10_swap_r2 (swap; same episode
seeds, so A-2 starts from the same position in each paired episode).
Terminal position = other_position at the last frame (t = 100).
"""

import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR, git_hash  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

OUT_DIR = os.path.join(OUT_ROOT, 'stageA', 'a2_terminal')
DATA = {
    'original': os.path.join(PI_DIR, 'data', 'data', 'r2_a1random_a2rl', 'data.h5'),
    'swap_rg': os.path.join(PI_DIR, 'data', 'data', 'v10_swap_r2', 'data.h5'),
}
TOP_LEFT = np.array([-9.0, 9.0])     # Red (original) / Green (swap)
BOTTOM_LEFT = np.array([-9.0, -9.0])  # Green (original) / Red (swap)
GREEN = {'original': BOTTOM_LEFT, 'swap_rg': TOP_LEFT}


def stats(pos):
    """pos: (N, T, 2)."""
    term = pos[:, -1]
    d_tl = np.linalg.norm(term - TOP_LEFT, axis=1)
    d_bl = np.linalg.norm(term - BOTTOM_LEFT, axis=1)
    all_tl = np.linalg.norm(pos - TOP_LEFT, axis=2)
    all_bl = np.linalg.norm(pos - BOTTOM_LEFT, axis=2)
    return {
        'n_episodes': int(len(term)),
        'terminal_mean_xy': term.mean(0).tolist(),
        'terminal_dist_to_top_left(-9,9)': {'mean': float(d_tl.mean()), 'median': float(np.median(d_tl))},
        'terminal_dist_to_bottom_left(-9,-9)': {'mean': float(d_bl.mean()), 'median': float(np.median(d_bl))},
        'frac_terminal_closer_to_top_left': float((d_tl < d_bl).mean()),
        'frac_terminal_within5_top_left': float((d_tl <= 5).mean()),
        'frac_terminal_within5_bottom_left': float((d_bl <= 5).mean()),
        'frac_frames_within5_top_left': float((all_tl <= 5).mean()),
        'frac_frames_within5_bottom_left': float((all_bl <= 5).mean()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pos = {}
    out = {'git_commit': git_hash(), 'terminal': 'other_position at t=100', 'splits': {}}
    for split in ('train', 'test'):
        out['splits'][split] = {}
        for lo, p in DATA.items():
            with h5py.File(p, 'r') as f:
                pos[(lo, split)] = f[f'{split}/other_position'][()]
            s = stats(pos[(lo, split)])
            g = GREEN[lo]
            s['green_here'] = g.tolist()
            out['splits'][split][lo] = s
        a, b = pos[('original', split)], pos[('swap_rg', split)]
        out['splits'][split]['paired'] = {
            'start_positions_identical': bool(np.array_equal(a[:, 0], b[:, 0])),
            'terminal_distance_original_vs_swap': {
                'mean': float(np.linalg.norm(a[:, -1] - b[:, -1], axis=1).mean()),
                'median': float(np.median(np.linalg.norm(a[:, -1] - b[:, -1], axis=1)))},
            'mean_framewise_distance': float(np.linalg.norm(a - b, axis=2).mean()),
        }
    with open(os.path.join(OUT_DIR, 'a2_terminal.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    lm = {'original': {'Red': TOP_LEFT, 'Green': BOTTOM_LEFT},
          'swap_rg': {'Red': BOTTOM_LEFT, 'Green': TOP_LEFT}}
    for ax, lo in zip(axes[:2], ('original', 'swap_rg')):
        term = pos[(lo, 'train')][:, -1]
        h = ax.hist2d(term[:, 0], term[:, 1], bins=np.linspace(-10, 10, 21), cmap='Blues')
        fig.colorbar(h[3], ax=ax, shrink=0.8, label='episodes')
        for nm, p in lm[lo].items():
            ax.scatter(*p, marker='D', s=90, c='red' if nm == 'Red' else 'limegreen', edgecolors='k')
        ax.set_title(f'A-2 terminal position (t=100), {lo}, train 3000 ep\n'
                     f'Green here at {tuple(GREEN[lo].astype(int))}', fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    ax = axes[2]
    for lo, c in (('original', '#1f77b4'), ('swap_rg', '#ff7f0e')):
        p = pos[(lo, 'train')]
        d = np.linalg.norm(p - BOTTOM_LEFT, axis=2).mean(0)
        d2 = np.linalg.norm(p - TOP_LEFT, axis=2).mean(0)
        ax.plot(d, color=c, lw=2, label=f'{lo}: to bottom-left (-9,-9)')
        ax.plot(d2, color=c, lw=2, ls='--', label=f'{lo}: to top-left (-9,9)')
    ax.set_xlabel('t')
    ax.set_ylabel('mean distance of A-2')
    ax.set_title('A-2 mean distance over the episode (train)', fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle('v10 stage A: A-2 actor (v3_rl_a2_actor.pth, trained in the original layout) in both layouts')
    fig.savefig(os.path.join(OUT_DIR, 'a2_terminal.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
