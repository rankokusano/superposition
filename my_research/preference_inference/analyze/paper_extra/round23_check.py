"""
Paper round 23 Sec.1 checks, data only (paper/check/paper_instructions_round23.md; docs/paper_extra_log.md Sec.47;
exploratory, outside the pre-registration). CPU, model independent.
  (2.1 data) round 20 data paper_r20_swap_p{50,80}: A-2's origin p50 = other_position[:, 50] (equal across N / O / S / R),
     X = A-2's liked colour in the pre-swap layout, Y = in the swapped layout; displacement 50 -> 80 per variant;
     fraction of episodes whose reference line p50 + d unit(L - p50) is inside the arena (|q| <= 10) for d = 0..5
     (the round-22 rule), per target.
  (2.2 ranges) round 21 data_noswap, A-2's view at frame 22 (T0 - 1 of the V1 flow): per range (8) the pixel counts of
     the four box colours (v11core.pleasure.color_mask); classes
       R  the range with the most red pixels;
       B  among ranges with blue or cyan pixels and no red / green pixel, the one with the most blue + cyan pixels;
       E  among ranges with no pixel of the four box colours, the one farthest (circular range distance) from the
          nearest range with a box pixel; ties -> more valid positive steps, then the lower index;
     direction = BIN_UNIT of the range (its centre); validity of p2 + d u, d = -5..5.
-> data/result/paper_extra/Round23/check/{check.json, ranges.npz}
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
from v11core.pleasure import color_mask, BIN_UNIT, N_BINS, COLS_PER_BIN  # noqa

DATA = os.path.join(_PI, 'data', 'data')
OUT = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round23', 'check')
D22 = np.arange(-5, 6)


def r20_targets(g):
    lay, lk = g['layout'][()], g['a2_liked'][()]
    col = np.where(lk == 1, 'Red', 'Green')
    X = np.array([L.LANDMARKS[L.CODE_LAYOUT[int(a)]][c] for a, c in zip(lay, col)], np.float64)
    Y = np.array([L.LANDMARKS[L.other_layout(L.CODE_LAYOUT[int(a)])][c] for a, c in zip(lay, col)], np.float64)
    return X, Y


def inside(p, u, ds):
    q = p[:, None] + ds[None, :, None] * u[:, None]
    return (np.abs(q) <= 10).all(-1)


def ranges(ov, p2):
    cnt = np.stack([color_mask(ov, c).reshape(ov.shape[0], -1, N_BINS, COLS_PER_BIN).sum((1, 3))
                    for c in ('Red', 'Green', 'Blue', 'Cyan')], -1)                 # (N, 8, 4)
    N = ov.shape[0]
    R = np.where(cnt[:, :, 0].max(1) > 0, cnt[:, :, 0].argmax(1), -1)
    bc = cnt[:, :, 2] + cnt[:, :, 3]
    okb = (bc > 0) & (cnt[:, :, 0] == 0) & (cnt[:, :, 1] == 0)
    B = np.where(okb.any(1), np.where(okb, bc, -1).argmax(1), -1)
    box = cnt.sum(-1) > 0
    E = np.full(N, -1)
    room = np.stack([inside(p2, np.repeat(BIN_UNIT[b][None], N, 0), np.arange(0, 6)).sum(1) for b in range(N_BINS)], 1)
    for n in range(N):
        emp = np.where(~box[n])[0]
        if len(emp) == 0 or not box[n].any():
            continue
        bx = np.where(box[n])[0]
        dist = np.array([np.min(np.minimum((e - bx) % N_BINS, (bx - e) % N_BINS)) for e in emp])
        key = [(-dist[i], -room[n, e], e) for i, e in enumerate(emp)]
        E[n] = emp[min(range(len(emp)), key=lambda i: key[i])]
    return {'R': R, 'B': B, 'E': E}, cnt


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {'r20': {}, 'r22dir': {}}
    for p in (50, 80):
        G = {v: h5py.File(os.path.join(DATA, f'paper_r20_swap_p{p}', f'data_{v}.h5'), 'r')['test'] for v in 'NOSR'}
        op = {v: G[v]['other_position'][()].astype(np.float64) for v in 'NOSR'}
        p50 = op['N'][:, 50]
        X, Y = r20_targets(G['N'])
        r = {'p50_equal_across_variants': bool(all(np.array_equal(op[v][:, 50], p50) for v in 'NOSR')),
             'A1_path_equal': bool(all(np.array_equal(G[v]['self_position'][()], G['N']['self_position'][()]) for v in 'OSR')),
             'targets_equal_across_variants': bool(all(np.array_equal(r20_targets(G[v])[0], X) for v in 'OSR'))}
        for v in 'NOSR':
            r[f'disp_50_80_mean|{v}'] = float(np.linalg.norm(op[v][:, 80] - p50, axis=1).mean())
            r[f'dist_end_X_median|{v}'] = float(np.median(np.linalg.norm(op[v][:, 80] - X, axis=1)))
            r[f'dist_end_Y_median|{v}'] = float(np.median(np.linalg.norm(op[v][:, 80] - Y, axis=1)))
        r['dist_p50_X_mean'] = float(np.linalg.norm(X - p50, axis=1).mean())
        r['dist_p50_Y_mean'] = float(np.linalg.norm(Y - p50, axis=1).mean())
        for nm, tg in (('X', X), ('Y', Y)):
            u = (tg - p50) / np.maximum(np.linalg.norm(tg - p50, axis=1, keepdims=True), 1e-9)
            r[f'valid_d0to5|{nm}'] = float(inside(p50, u, np.arange(0, 6)).all(1).mean())
        uX = (X - p50) / np.linalg.norm(X - p50, axis=1, keepdims=True)
        uY = (Y - p50) / np.linalg.norm(Y - p50, axis=1, keepdims=True)
        r['valid_d0to5_both'] = float((inside(p50, uX, np.arange(0, 6)).all(1) & inside(p50, uY, np.arange(0, 6)).all(1)).mean())
        out['r20'][f'p{p}'] = r
    g = h5py.File(os.path.join(DATA, 'paper_r21_swap', 'data_noswap.h5'), 'r')['test']
    ov = g['other_vision'][:, 22]
    p2 = g['other_position'][:, 0].astype(np.float64)
    cls, cnt = ranges(ov, p2)
    d = {}
    for c, b in cls.items():
        has = b >= 0
        d[f'available|{c}'] = float(has.mean())
        d[f'range_counts|{c}'] = np.bincount(b[has], minlength=8).tolist()
        u = BIN_UNIT[np.maximum(b, 0)]
        v = inside(p2, u, D22)
        d[f'valid_d-5to5|{c}'] = float((v.all(1) & has).mean())
        d[f'valid_d0to5|{c}'] = float((v[:, 5:].all(1) & has).mean())
        d[f'valid_d-3to3|{c}'] = float((v[:, 2:9].all(1) & has).mean())
        d[f'valid_d0to3|{c}'] = float((v[:, 5:9].all(1) & has).mean())
        d[f'valid_per_d|{c}'] = (v & has[:, None]).mean(0).round(4).tolist()
    d['all_three_available'] = float(((cls['R'] >= 0) & (cls['B'] >= 0) & (cls['E'] >= 0)).mean())
    d['n_box_ranges_mean'] = float((cnt.sum(-1) > 0).sum(1).mean())
    d['n_empty_ranges_mean'] = float((cnt.sum(-1) == 0).sum(1).mean())
    out['r22dir'] = d
    np.savez(os.path.join(OUT, 'ranges.npz'), R=cls['R'], B=cls['B'], E=cls['E'], counts=cnt)
    json.dump(out, open(os.path.join(OUT, 'check.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
