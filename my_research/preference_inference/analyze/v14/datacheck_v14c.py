"""
v14-C data check (docs/v14_experiment_log.md Sec.2.5, 17), as v13's gate61_v13 items 1, 2, 4 for v14_s1_mixedpref.
Reads only; writes data/result/v14/datacheck/datacheck_v14c.json.
  1 reproducibility: check_mixedpref_a == _b == first 3 train episodes of the full data
  2 pairing with v11_s1_random: layout, initial A-1 position, A-2 positions (all t), sleep frames equal (train, test);
    Red-liked episodes identical to v11_s1_mixed in all arrays; rule turns (sources 1 + 2) at the same steps
  3 a1_liked: share Red per split, independence from layout (2 x 2 counts), a1_liked_t == a1_liked repeated
  4 behaviour: time within 5 of the LIKED colour's reference point and of the disliked one, per liked colour and
    layout, next to v11_s1_mixed (liked Red) and v11_s1_random (distance to Red / Green)
"""

import json
import os
import sys

import h5py
import numpy as np

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI)
from sim_v10 import layout as L  # noqa

D = os.path.join(_PI, 'data', 'data')
OUT = os.path.join(_PI, 'data', 'result', 'v14', 'datacheck')
KEYS = ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_motion', 'other_position', 'a1_source', 'layout')
REF = {c: {0: np.array(L.LANDMARKS[L.ORIGINAL][c]), 1: np.array(L.LANDMARKS[L.SWAP_RG][c])} for c in ('Red', 'Green')}


def h5(name):
    return h5py.File(os.path.join(D, name, 'data.h5'), 'r')


def near(pos, lay, colour):
    ref = np.stack([REF[colour][int(l)] for l in lay])[:, None, :]
    return np.linalg.norm(pos - ref, axis=-1) < 5


def main():
    os.makedirs(OUT, exist_ok=True)
    F, R, M = h5('v14_s1_mixedpref'), h5('v11_s1_random'), h5('v11_s1_mixed')
    res = {}
    a = h5py.File(os.path.join(OUT, 'check_mixedpref_a.h5'), 'r')['train']
    b = h5py.File(os.path.join(OUT, 'check_mixedpref_b.h5'), 'r')['train']
    res['1_a_eq_b'] = all(np.array_equal(a[k][()], b[k][()]) for k in KEYS + ('a1_liked',))
    res['1_a_eq_full_first3'] = all(np.array_equal(a[k][()], F['train'][k][:3]) for k in KEYS + ('a1_liked',))
    for sp in ('train', 'test'):
        f, r, m = F[sp], R[sp], M[sp]
        lk = f['a1_liked'][()]
        n = len(lk)
        r2 = {'n': n, 'share_red': float(lk.mean()),
              'liked_t_ok': bool((f['a1_liked_t'][()] == lk[:, None]).all()),
              'layout_eq': bool(np.array_equal(f['layout'][()], r['layout'][()])),
              'init_pos_eq': bool(np.array_equal(f['self_position'][:, 0], r['self_position'][:, 0])),
              'a2_pos_eq': bool(np.array_equal(f['other_position'][()], r['other_position'][()]))}
        sf, sr, sm = f['a1_source'][()], r['a1_source'][()], m['a1_source'][()]
        r2['sleep_eq'] = bool(np.array_equal(sf == 3, sr == 3))
        r2['rule_turns_eq_mixed'] = bool(np.array_equal(np.isin(sf, [1, 2]), np.isin(sm, [1, 2])))
        red = np.where(lk == 1)[0]
        same = []
        for i0 in range(0, len(red), 100):
            idx = red[i0:i0 + 100]
            same.append(all(np.array_equal(f[k][idx], m[k][idx]) for k in KEYS if k != 'layout'))
        r2['red_eps_identical_to_mixed'] = bool(all(same))
        lay = f['layout'][()]
        r2['liked_x_layout_counts'] = {f'liked{c}_layout{l}': int(((lk == c) & (lay == l)).sum()) for c in (0, 1) for l in (0, 1)}
        pos = f['self_position'][()]
        nl = np.where(lk[:, None] == 1, near(pos, lay, 'Red'), near(pos, lay, 'Green'))
        nd = np.where(lk[:, None] == 1, near(pos, lay, 'Green'), near(pos, lay, 'Red'))
        r2['near_liked5'] = {f'liked_{nm}': float(nl[lk == c].mean()) for c, nm in ((1, 'Red'), (0, 'Green'))}
        r2['near_disliked5'] = {f'liked_{nm}': float(nd[lk == c].mean()) for c, nm in ((1, 'Red'), (0, 'Green'))}
        r2['near_liked5_by_layout'] = {f'liked_{nm}_layout{l}': float(nl[(lk == c) & (lay == l)].mean())
                                       for c, nm in ((1, 'Red'), (0, 'Green')) for l in (0, 1)}
        aw = sf != 3
        r2['rule_share_awake'] = float(np.isin(sf, [1, 2])[aw].mean())
        r2['v11_mixed_near_red5'] = float(near(m['self_position'][()], m['layout'][()], 'Red').mean())
        r2['v11_random_near_red5'] = float(near(r['self_position'][()], r['layout'][()], 'Red').mean())
        r2['v11_random_near_green5'] = float(near(r['self_position'][()], r['layout'][()], 'Green').mean())
        res['2_' + sp] = r2
    res['pass'] = res['1_a_eq_b'] and res['1_a_eq_full_first3'] and all(
        res['2_' + sp][k] for sp in ('train', 'test') for k in ('liked_t_ok', 'layout_eq', 'init_pos_eq', 'a2_pos_eq',
                                                                 'sleep_eq', 'rule_turns_eq_mixed', 'red_eps_identical_to_mixed'))
    json.dump(res, open(os.path.join(OUT, 'datacheck_v14c.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
