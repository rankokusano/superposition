"""
Paper round 21 Sec.1.2 checks 1-2 (docs/paper_extra_log.md Sec.45; criteria fixed before reading the data). CPU.
  1. frames t < T_S (= 20) bit-identical between data_swap and data_noswap (self / other vision, motions, positions,
     a1_source, layout_t); also reported: A-1's path and A-2's position identical over all frames (expected by the
     design; needed so that the swap / no-swap difference removes the path).
  2. layout_t flipped from frame T_S in swap only (constant in noswap); at frame T_S the red mask of swap equals the
     green mask of noswap and vice versa (v11core.pleasure.color_mask), for A-1's and A-2's views; images
     (swap_images.png: 4 episodes, A-1 view at frames T_S-1 / T_S (noswap / swap) and T_S+3 (swap), with a map of the
     positions and L_old / L_new) are looked at by eye.
-> data/result/paper_extra/Round21/check/{data_check.json, swap_images.png}
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
from sim_v10 import layout as L  # noqa
from v11core.pleasure import color_mask  # noqa

DATA = os.path.join(_PI, 'data', 'data', 'paper_r21_swap')
OUT = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round21', 'check')
T_S = 20
KEYS = ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_motion', 'other_position', 'a1_source',
        'layout_t')


def targets(lay):
    lo = np.array([L.LANDMARKS[L.CODE_LAYOUT[int(c)]]['Red'] for c in lay], np.float64)
    ln = np.array([L.LANDMARKS[L.other_layout(L.CODE_LAYOUT[int(c)])]['Red'] for c in lay], np.float64)
    return lo, ln


def main():
    os.makedirs(OUT, exist_ok=True)
    D = {v: {k: x[()] for k, x in h5py.File(os.path.join(DATA, f'data_{v}.h5'), 'r')['test'].items()}
         for v in ('swap', 'noswap')}
    a, b = D['swap'], D['noswap']
    N = a['layout'].shape[0]
    R = {'n': int(N), 'layout_counts': np.bincount(a['layout'], minlength=2).tolist(),
         'layout_equal': bool(np.array_equal(a['layout'], b['layout']))}
    c1 = {k: bool(np.array_equal(a[k][:, :T_S], b[k][:, :T_S])) for k in KEYS}
    c1.update({'A1_path_all_frames': bool(np.array_equal(a['self_position'], b['self_position']) and
                                          np.array_equal(a['self_motion'], b['self_motion'])),
               'A2_position_all_frames': bool(np.array_equal(a['other_position'], b['other_position'])),
               'A2_static': bool((a['other_position'] == a['other_position'][:, :1]).all())})
    R['check1'] = c1
    R['check1_pass'] = bool(all(c1[k] for k in KEYS) and R['layout_equal'])
    lay0 = a['layout']
    c2 = {'layout_flipped_swap': bool((a['layout_t'][:, T_S:] == 1 - lay0[:, None]).all() and
                                      (a['layout_t'][:, :T_S] == lay0[:, None]).all()),
          'layout_constant_noswap': bool((b['layout_t'] == lay0[:, None]).all())}
    for cam in ('self_vision', 'other_vision'):
        x, y = a[cam][:, T_S], b[cam][:, T_S]
        ex = (color_mask(x, 'Red') == color_mask(y, 'Green')).all((1, 2)) & (color_mask(x, 'Green') == color_mask(y, 'Red')).all((1, 2))
        vis = color_mask(y, 'Red').any((1, 2)) | color_mask(y, 'Green').any((1, 2))
        c2[f'{cam}_masks_exchanged_n'] = int(ex.sum())
        c2[f'{cam}_red_or_green_visible_n'] = int(vis.sum())
    R['check2'] = c2
    R['check2_pass'] = bool(c2['layout_flipped_swap'] and c2['layout_constant_noswap'] and
                            c2['self_vision_masks_exchanged_n'] == N and c2['other_vision_masks_exchanged_n'] == N)
    lo, ln = targets(lay0)
    p = a['self_position'].astype(np.float64)
    R['describe'] = {'A1_dist_L_old_at_23_mean': float(np.linalg.norm(p[:, 23] - lo, axis=1).mean()),
                     'A1_dist_L_new_at_23_mean': float(np.linalg.norm(p[:, 23] - ln, axis=1).mean()),
                     'A2_dist_L_old_mean': float(np.linalg.norm(a['other_position'][:, 0] - lo, axis=1).mean()),
                     'A2_dist_L_new_mean': float(np.linalg.norm(a['other_position'][:, 0] - ln, axis=1).mean()),
                     'L_old_L_new_dist': float(np.linalg.norm(lo - ln, axis=1).mean())}
    json.dump(R, open(os.path.join(OUT, 'data_check.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    eps = [0, 1, 2, 3]
    fig, axs = plt.subplots(len(eps), 5, figsize=(17, 2.0 * len(eps)), gridspec_kw={'width_ratios': [3, 3, 3, 3, 1.3]})
    for i, n in enumerate(eps):
        for j, (v, t, cam) in enumerate((('noswap', T_S - 1, 'self_vision'), ('noswap', T_S, 'self_vision'),
                                         ('swap', T_S, 'self_vision'), ('swap', T_S, 'other_vision'))):
            axs[i, j].imshow(D[v][cam][n, t], interpolation='nearest', aspect='auto')
            axs[i, j].set_title(f'ep{n} {v} frame {t} {"A-2" if cam == "other_vision" else "A-1"} view', fontsize=8)
            axs[i, j].axis('off')
        ax = axs[i, 4]
        ax.plot(p[n, :24, 0], p[n, :24, 1], color='C0', lw=1)
        ax.plot(*p[n, 23], 'o', color='C0', ms=4)
        ax.plot(*a['other_position'][n, 0], 'ko', ms=4)
        ax.plot(*lo[n], 'rs', ms=6); ax.plot(*ln[n], 'r^', ms=6)
        ax.set_xlim(-10.5, 10.5); ax.set_ylim(-10.5, 10.5); ax.set_aspect('equal'); ax.tick_params(labelsize=6)
        ax.set_title('A-1 path to t=23, A-2 (black); red sq L_old, tri L_new', fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'swap_images.png'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
