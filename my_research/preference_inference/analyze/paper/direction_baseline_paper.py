"""
Paper round 1 (paper/check/paper_round1_log.md Sec.1; descriptive, not pre-registered, data only, CPU):
balanced accuracy of the "direction only" baseline of experiment 4 (v14-A) on the pref test.
  direction choice: c+ if A-2's displacement over t = 1..T (pos[T+1] - pos[1]; T = 100 uses pos[100] - pos[1])
                    projects further on the bins-6,7 direction than on the bins-0,1 direction, else c-
                    (v14 log Sec.18, same code as analyze/v14/fp_direction_v14.py);
  answer: correct_sign (v14 log Sec.3 C); BA over the layout x liked cells with the v14-A bootstrap.
Also the raw agreement and the lowest third of A-2's rule-step share (v14 log Sec.18).
Writes paper/check/direction_baseline.json.
"""

import glob
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v14'))
from aggregate_inverse_v14 import BASE, TS, ba, boot, cells, ci  # noqa
from eval_s2_v11 import load_set  # noqa
from v11core.pleasure import BIN_UNIT  # noqa

OUT = os.path.join(_PI, 'paper', 'check', 'direction_baseline.json')


def direction(pos, T):
    disp = pos[:, min(T + 1, pos.shape[1] - 1)] - pos[:, 1]
    u01, u67 = BIN_UNIT[[0, 1]].mean(0), BIN_UNIT[[6, 7]].mean(0)
    u01, u67 = u01 / np.linalg.norm(u01), u67 / np.linalg.norm(u67)
    return np.where(disp @ u67 > disp @ u01, 1, -1)


def summary(cor, cl, idx):
    c, k = cor[idx], cl[idx]
    B = boot(k, len(idx))
    return {'n': int(len(idx)), 'ba': ba(c, k), 'ba_ci': ci([ba(c, k, b) for b in B]),
            'raw_agree': float(c.mean()), 'raw_agree_ci': ci([c[b].mean() for b in B]),
            'cell_acc': {int(j): float(c[k == j].mean()) for j in np.unique(k)},
            'cell_n': {int(j): int((k == j).sum()) for j in np.unique(k)}}


def main():
    d = load_set('pref')
    g = h5py.File(d['path'], 'r')[d['split']]
    pos = d['other_position']
    # the data-only fields are identical in every saved v14-A output; take one and check the rest
    ref = np.load(os.path.join(BASE, 'inverse', 'v12_mixed_0_ep200_pref.npz'))
    pref, cl = ref['correct_sign'], cells(ref)
    n_checked = 0
    for f in sorted(glob.glob(os.path.join(BASE, 'inverse', '*_pref.npz'))):
        z = np.load(f)
        assert (z['correct_sign'] == pref).all() and (cells(z) == cl).all(), f
        n_checked += 1
    assert (np.asarray(d['layout']).astype(int) * 2 + np.asarray(d['a2_liked']).astype(int) == cl).all()
    src = g['a2_source'][:, 1:51]
    aw = src != 3
    rule = ((src == 1) & aw).sum(1) / np.maximum(aw.sum(1), 1)
    low = np.where(rule <= np.percentile(rule, 100 / 3))[0]
    allep = np.arange(len(pref))
    out = {'definition': 'paper_round1_log.md Sec.1', 'n_npz_checked': n_checked,
           'rule_share_low_third_max': float(np.percentile(rule, 100 / 3)), 'n_low': int(len(low))}
    for T in TS:
        dirn = direction(pos, T)
        cor = (dirn == pref).astype(float)   # the direction choice is correct when it equals the preference answer
        for nm, idx in (('all', allep), ('low_rule', low)):
            out[f'T{T}|{nm}'] = summary(cor, cl, idx)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('checked npz', n_checked, '| low third rule share <=', round(out['rule_share_low_third_max'], 3),
          'n', out['n_low'])
    for nm in ('all', 'low_rule'):
        for T in TS:
            r = out[f'T{T}|{nm}']
            print(f"{nm:8s} T{T:<3d} BA {r['ba']:.3f} {np.round(r['ba_ci'], 3).tolist()}  raw {r['raw_agree']:.3f}  "
                  f"cells {np.round(list(r['cell_acc'].values()), 3).tolist()}")


if __name__ == '__main__':
    main()
