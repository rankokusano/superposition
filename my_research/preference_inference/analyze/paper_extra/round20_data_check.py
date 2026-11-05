"""
Paper round 20 Sec.1.2 checks 1-3 (paper/check/paper_instructions_round20.md; docs/paper_extra_log.md Sec.44;
exploratory, outside the pre-registration). Data only (no model), CPU.
  1. first half identical: N, O, S, R, frames 0..T_S-1, bit for bit: self/other vision, positions, motions, sources,
     layout_t, A-1's pleasure (Red - Green from A-1's view) and A-2's true pleasure (liked - disliked from A-2's view);
     A-1's path (self_position, self_motion, a1_source) identical over all frames;
  2. swap: layout_t at t = T_S flipped in N / O only; at frame T_S the red mask of N equals the green mask of S and
     vice versa (A-1's and A-2's views; positions at frame T_S are the same); images of a few episodes;
  3. destinations: at the last frame (80) the distance of A-2 to the new location of the liked colour (Y = the liked
     colour's reference point after the swap) and to the old location (X = before the swap); stop condition (fixed
     before reading the data): median dY(N) < median dY(O) and median dX(O) < median dX(N) and the share of episodes
     closer to their own destination (N: dY < dX, O: dX < dY) > 0.5 for both, at each p_rule. Also: A-2's travelled
     distance in the second half N vs O, O == S and N == R positions (by construction, described).
-> data/result/paper_extra/Round20/check/{data_check.json, swap_images.png, dist_last.png}
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
from sim_v10 import layout  # noqa
from v11core.pleasure import pleasure_np  # noqa

DATA = os.path.join(_PI, 'data', 'data')
OUT = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round20', 'check')
PS = (80, 50)
VARS = ('N', 'O', 'S', 'R')
T_S, LAST = 50, 80
KS = ('self_vision', 'other_vision', 'self_position', 'other_position', 'self_motion', 'other_motion',
      'a1_source', 'a2_source', 'layout_t')


def path(p, v):
    return os.path.join(DATA, f'paper_r20_swap_p{p}', f'data_{v}.h5')


def landmarks(lay0, liked):
    """X (old location of the liked colour) and Y (new location) per episode."""
    lc = np.where(liked == 1, 'Red', 'Green')
    X = np.array([layout.LANDMARKS[layout.CODE_LAYOUT[int(L)]][c] for L, c in zip(lay0, lc)])
    Y = np.array([layout.LANDMARKS[layout.other_layout(layout.CODE_LAYOUT[int(L)])][c] for L, c in zip(lay0, lc)])
    return X, Y


def a2_pleasure(ov, liked):
    out = np.empty(ov.shape[:2] + (8,), np.float32)
    for lk in (0, 1):
        m = liked == lk
        a, b = ('Red', 'Green') if lk == 1 else ('Green', 'Red')
        out[m] = pleasure_np(ov[m], a, b)
    return out


def check_p(p):
    G = {v: h5py.File(path(p, v), 'r')['test'] for v in VARS}
    D = {v: {k: G[v][k][()] for k in KS} for v in VARS}
    lay0, liked = G['N']['layout'][()], G['N']['a2_liked'][()]
    R = {'n': int(len(lay0)), 'cells': {f"{'red' if lk else 'green'}|L{L}": int(((liked == lk) & (lay0 == L)).sum())
                                         for lk in (1, 0) for L in (0, 1)}}
    for v in VARS[1:]:
        assert np.array_equal(G[v]['layout'][()], lay0) and np.array_equal(G[v]['a2_liked'][()], liked)
    # 1. first half
    P1 = {v: pleasure_np(D[v]['self_vision'][:, :T_S], 'Red', 'Green') for v in VARS}
    P2 = {v: a2_pleasure(D[v]['other_vision'][:, :T_S], liked) for v in VARS}
    c1 = {}
    for v in VARS[1:]:
        r = {k: bool(np.array_equal(D[v][k][:, :T_S], D['N'][k][:, :T_S])) for k in KS}
        r['a1_pleasure'] = bool(np.array_equal(P1[v], P1['N']))
        r['a2_true_pleasure'] = bool(np.array_equal(P2[v], P2['N']))
        r['positions_frame_T_S'] = bool(np.array_equal(D[v]['self_position'][:, T_S], D['N']['self_position'][:, T_S]) and
                                        np.array_equal(D[v]['other_position'][:, T_S], D['N']['other_position'][:, T_S]))
        r['a1_path_all_frames'] = bool(all(np.array_equal(D[v][k], D['N'][k]) for k in ('self_position', 'self_motion', 'a1_source')))
        c1[f'{v}_vs_N'] = r
    R['check1'] = c1
    R['check1_pass'] = all(all(r.values()) for r in c1.values())
    # 2. swap
    lt = {v: D[v]['layout_t'] for v in VARS}
    c2 = {'layout_flipped_N_O': bool(all((lt[v][:, T_S:] == 1 - lay0[:, None]).all() and (lt[v][:, :T_S] == lay0[:, None]).all() for v in 'NO')),
          'layout_constant_S_R': bool(all((lt[v] == lay0[:, None]).all() for v in 'SR'))}
    for cam in ('self_vision', 'other_vision'):
        from v11core.pleasure import color_mask
        a, b = D['N'][cam][:, T_S], D['S'][cam][:, T_S]
        ex = (color_mask(a, 'Red') == color_mask(b, 'Green')).all((1, 2)) & (color_mask(a, 'Green') == color_mask(b, 'Red')).all((1, 2))
        vis = color_mask(b, 'Red').any((1, 2)) | color_mask(b, 'Green').any((1, 2))
        c2[f'{cam}_masks_exchanged_n'] = int(ex.sum())
        c2[f'{cam}_red_or_green_visible_n'] = int(vis.sum())
        c2[f'{cam}_frame_T_S_minus_1_same_N_S'] = bool(np.array_equal(D['N'][cam][:, T_S - 1], D['S'][cam][:, T_S - 1]))
    R['check2'] = c2
    R['check2_pass'] = bool(c2['layout_flipped_N_O'] and c2['layout_constant_S_R'] and
                            c2['self_vision_masks_exchanged_n'] == R['n'] and c2['other_vision_masks_exchanged_n'] == R['n'])
    # 3. destinations
    X, Y = landmarks(lay0, liked)
    c3 = {}
    for v in VARS:
        op = D[v]['other_position'].astype(np.float64)
        dX, dY = np.linalg.norm(op[:, LAST] - X, axis=1), np.linalg.norm(op[:, LAST] - Y, axis=1)
        c3[v] = {'dX_last_median': float(np.median(dX)), 'dY_last_median': float(np.median(dY)),
                 'dX_last_mean': float(dX.mean()), 'dY_last_mean': float(dY.mean()),
                 'dX_T_S_mean': float(np.linalg.norm(op[:, T_S] - X, axis=1).mean()),
                 'dY_T_S_mean': float(np.linalg.norm(op[:, T_S] - Y, axis=1).mean()),
                 'share_closer_to_Y': float((dY < dX).mean()), 'share_closer_to_X': float((dX < dY).mean()),
                 'path_len_2nd_half_mean': float(np.linalg.norm(np.diff(op[:, T_S:], axis=1), axis=2).sum(1).mean()),
                 'a2_source_2nd_half_share': (np.bincount(D[v]['a2_source'][:, T_S:LAST].ravel(), minlength=4) /
                                              D[v]['a2_source'][:, T_S:LAST].size).tolist()}
        c3[v]['_dX'], c3[v]['_dY'] = dX, dY
    pl = {v: np.linalg.norm(np.diff(D[v]['other_position'][:, T_S:].astype(np.float64), axis=1), axis=2).sum(1) for v in VARS}
    c3['path_len_N_minus_O'] = {'max_abs': float(np.abs(pl['N'] - pl['O']).max()), 'mean': float((pl['N'] - pl['O']).mean()),
                                'n_differ': int((np.abs(pl['N'] - pl['O']) > 1e-4).sum())}
    c3['O_equals_S_positions'] = bool(np.array_equal(D['O']['other_position'], D['S']['other_position']))
    c3['N_equals_R_positions'] = bool(np.array_equal(D['N']['other_position'], D['R']['other_position']))
    c3['pass'] = bool(c3['N']['dY_last_median'] < c3['O']['dY_last_median'] and
                      c3['O']['dX_last_median'] < c3['N']['dX_last_median'] and
                      c3['N']['share_closer_to_Y'] > 0.5 and c3['O']['share_closer_to_X'] > 0.5)
    R['check3'] = c3
    R['check3_pass'] = c3['pass']
    return R, D, X, Y


def figures(res):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    R, D, X, Y = res[80]
    # swap images: 3 episodes, A-1 and A-2 views at frames T_S-1, T_S for N and S, and the A-2 path (N, O)
    eps = [0, 1, 2]
    fig, axs = plt.subplots(len(eps), 5, figsize=(17, 2.0 * len(eps)), gridspec_kw={'width_ratios': [3, 3, 3, 3, 1.3]})
    for i, n in enumerate(eps):
        for j, (v, t, cam) in enumerate((('S', T_S - 1, 'other_vision'), ('N', T_S, 'other_vision'),
                                         ('S', T_S, 'other_vision'), ('N', T_S, 'self_vision'))):
            axs[i, j].imshow(D[v][cam][n, t], interpolation='nearest', aspect='auto')
            axs[i, j].set_title(f'ep{n} {v} frame {t} {"A-2" if cam == "other_vision" else "A-1"} view', fontsize=8)
            axs[i, j].axis('off')
        ax = axs[i, 4]
        for v, c in (('N', 'C0'), ('O', 'C1')):
            op = D[v]['other_position'][n]
            ax.plot(op[:T_S + 1, 0], op[:T_S + 1, 1], color='0.6', lw=0.8)
            ax.plot(op[T_S:, 0], op[T_S:, 1], color=c, lw=1.2, label=v)
        ax.plot(*X[n], 'ks', ms=5); ax.plot(*Y[n], 'k^', ms=5)
        ax.set_xlim(-10.5, 10.5); ax.set_ylim(-10.5, 10.5); ax.set_aspect('equal'); ax.tick_params(labelsize=6)
        ax.set_title('A-2 path (sq old, tri new)', fontsize=7)
        if i == 0:
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'swap_images.png'), dpi=130)
    plt.close(fig)
    # last-frame distances
    fig, axs = plt.subplots(1, 4, figsize=(14, 3))
    bins = np.linspace(0, 26, 27)
    for k, p in enumerate(PS):
        c3 = res[p][0]['check3']
        for j, (nm, key) in enumerate((('to new location (Y)', '_dY'), ('to old location (X)', '_dX'))):
            ax = axs[2 * k + j]
            for v, c in (('N', 'C0'), ('O', 'C1')):
                ax.hist(c3[v][key], bins=bins, histtype='step', color=c, lw=1.5, label=f'{v} (median {np.median(c3[v][key]):.1f})')
            ax.set_title(f'p_rule 0.{p // 10}: A-2 distance {nm}, frame {LAST}', fontsize=8)
            ax.legend(fontsize=7); ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'dist_last.png'), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {p: check_p(p) for p in PS}
    figures(res)
    out = {}
    for p in PS:
        R = res[p][0]
        for v in VARS:
            R['check3'][v] = {k: x for k, x in R['check3'][v].items() if not k.startswith('_')}
        out[f'p{p}'] = R
    out['all_pass'] = all(out[f'p{p}'][f'check{i}_pass'] for p in PS for i in (1, 2, 3))
    json.dump(out, open(os.path.join(OUT, 'data_check.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({f'p{p}': {k: out[f'p{p}'][k] for k in ('n', 'cells', 'check1_pass', 'check2', 'check2_pass', 'check3_pass')}
                      for p in PS}, ensure_ascii=False, indent=1))
    print(json.dumps({f'p{p}': out[f'p{p}']['check3'] for p in PS}, indent=1))
    print('all_pass', out['all_pass'])


if __name__ == '__main__':
    main()
