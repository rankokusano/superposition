"""
Paper round 1 (paper/check/paper_round1_log.md Sec.3; descriptive, saved outputs only, CPU): share of choices
following A-2's preference among the 88 episodes where preference and A-2's direction over t = 1..50 disagree,
for every late-5 checkpoint of the l1 stage and of the mse stage, with analyze/v14/fp_direction_v14.stats applied
to the saved v14-A outputs (data/result/v14/inverse/<tag>_ep<e>_pref.npz), T = 50. Checks the overlap with the
existing fp_direction_*.json files. Writes paper/check/follow_pref_late5.json.
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
for d in ('v11', 'v14'):
    sys.path.insert(0, os.path.join(_PI, 'analyze', d))
from aggregate_inverse_v14 import BASE, EPOCHS, choice  # noqa
from eval_s2_v11 import load_set  # noqa
from fp_direction_v14 import T, stats  # noqa
from direction_baseline_paper import direction  # noqa

SUB = os.path.join(BASE, 'inverse')
OUT = os.path.join(_PI, 'paper', 'check', 'follow_pref_late5.json')
TAGS = ([f'v13_mixed_{s}' for s in range(6)] + [f'v12_mixed_{s}' for s in range(6)] +
        [f'v12_random_{s}' for s in range(6)] + [f'v13ns_mixed_{s}' for s in range(3)] +
        [f'v13mse_mixed_{s}' for s in range(6)] + [f'v12mse_mixed_{s}' for s in (0, 1, 2, 4, 5)] +
        [f'v12mse_random_{s}' for s in range(6)] + [f'v13nsmse_mixed_{s}' for s in range(3)])
# existing files: key in our output -> (file, key in that file)
CHECK = []
for s in range(3):
    CHECK += [(f'v13mse_mixed_{s}|FP|all|ep{e}', 'fp_direction_v13mse.json', f'v13mse_mixed_{s}|FP|all|ep{e}') for e in EPOCHS]
    CHECK += [(f'v13_mixed_{s}|L1|all|ep200', 'fp_direction_v13l1.json', f'v13_mixed_{s}|L1|all|ep200')]
    CHECK += [(f'v13nsmse_mixed_{s}|FP|all|ep{e}', 'fp_direction_nsmse.json', f'v13nsmse_mixed_{s}|FP|all|ep{e}') for e in EPOCHS]
    CHECK += [(f'v13ns_mixed_{s}|L1|all|ep{e}', 'fp_direction_nsl1.json', f'v13ns_mixed_{s}|L1|all|ep{e}') for e in EPOCHS]
for s in range(4):
    CHECK += [(f'v12_mixed_{s}|{q}|all|ep{e}', 'fp_direction.json', f'mixed_s{s}|{q}|all|ep{e}') for e in EPOCHS for q in ('FP', 'L1')]
CHECK += [(f'v13mse_mixed_3|FP|all|ep{e}', 'fp_direction_v13mse_s3.json', f'v13mse_mixed_3|FP|all|ep{e}') for e in EPOCHS]
CHECK += [(f'v13mse_mixed_{s}|FP|all|ep{e}', 'fp_direction_mse_s45.json', f'v13mse_mixed_{s}|FP|all|ep{e}') for e in EPOCHS for s in (4, 5)]


def main():
    d = load_set('pref')
    g = h5py.File(d['path'], 'r')[d['split']]
    dirn = direction(d['other_position'], T)
    src = g['a2_source'][:, 1:51]
    aw = src != 3
    rule = ((src == 1) & aw).sum(1) / np.maximum(aw.sum(1), 1)
    low = np.where(rule <= np.percentile(rule, 100 / 3))[0]
    allep = np.arange(len(rule))
    out = {}
    for tag in TAGS:
        for e in EPOCHS:
            z = np.load(os.path.join(SUB, f'{tag}_ep{e}_pref.npz'))
            for q in ('FP', 'L1'):
                chp = choice(z, q, T)
                for nm, idx in (('all', allep), ('low_rule', low)):
                    out[f'{tag}|{q}|{nm}|ep{e}'] = stats(chp, z['correct_sign'], dirn, idx, np.random.RandomState(0))
    bad = 0
    for k, f, fk in CHECK:
        ref = json.load(open(os.path.join(SUB, f)))[fk]
        same = all(np.allclose(np.atleast_1d(out[k][m]), np.atleast_1d(ref[m])) for m in ref)
        bad += not same
        if not same:
            print('MISMATCH', k, f, fk)
    print(f'checked {len(CHECK)} overlapping entries, mismatches {bad}')
    summ = {}
    for tag in TAGS:
        for q in ('FP', 'L1'):
            rows = [out[f'{tag}|{q}|all|ep{e}'] for e in EPOCHS]
            v = [r['disagree_follow_pref'] for r in rows]
            summ[f'{tag}|{q}'] = {'late5_mean': float(np.mean(v)), 'late5_min': float(min(v)), 'late5_max': float(max(v)),
                                  'n_ci_low_above_half': int(sum(r['disagree_follow_pref_ci'][0] > 0.5 for r in rows)),
                                  'ep200': v[-1], 'ep200_ci': rows[-1]['disagree_follow_pref_ci'],
                                  'n_disagree': rows[-1]['n_disagree']}
    json.dump({'definition': 'paper_round1_log.md Sec.3', 'n_check': len(CHECK), 'n_mismatch': bad,
               'summary': summ, 'detail': out}, open(OUT, 'w'), indent=1)
    for k, r in summ.items():
        print(f"{k:22s} {r['late5_min']:.3f}-{r['late5_max']:.3f}  CI>0.5 {r['n_ci_low_above_half']}/5")


if __name__ == '__main__':
    main()
