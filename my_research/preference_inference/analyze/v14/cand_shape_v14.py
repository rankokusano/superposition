"""
v14-A candidate shape (docs/v14_instructions.md Sec.1.1, docs/v14_experiment_log.md).
Per-bin mean |P| of the innate pleasure (liked Red, disliked Green, v11core.pleasure) over the stage-1
training data seen by the v12 models (v11_s1_mixed and v11_s1_random train, 3000 ep x t = 0..99 each),
pooled over both conditions, then mirrored (bin k <-> bin 7-k averaged) so that c+ and c- differ only
in which side is pleasant. Bins 2-5 are always 0 in stage 1.
Candidate c(u) = u * [-m0, -m1, 0, 0, 0, 0, m6, m7] (the 2nd-stage-b 'signed_pairs' layout with u fixed).
Writes data/result/v14/cand_shape.json (also the per-condition, unmirrored values).
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI)
from v11core.pleasure import pleasure_np  # noqa

DATA = os.path.join(_PI, 'data', 'data')
OUT = os.path.join(_PI, 'data', 'result', 'v14')


def mean_abs(name, chunk=100):
    g = h5py.File(os.path.join(DATA, name, 'data.h5'), 'r')['train']
    N = g['layout'].shape[0]
    S, n = np.zeros(8), 0
    for b in range(0, N, chunk):
        P = pleasure_np(g['self_vision'][b:b + chunk, :100])
        S += np.abs(P).sum((0, 1))
        n += P.shape[0] * P.shape[1]
    return S / n


def main():
    os.makedirs(OUT, exist_ok=True)
    per = {c: mean_abs(f'v11_s1_{c}') for c in ('mixed', 'random')}
    pooled = (per['mixed'] + per['random']) / 2
    m = (pooled + pooled[::-1]) / 2
    shape = np.array([-m[0], -m[1], 0, 0, 0, 0, m[6], m[7]])
    r = {'per_condition_mean_abs_P': {c: v.tolist() for c, v in per.items()},
         'pooled_mean_abs_P': pooled.tolist(), 'mirrored_m': m.tolist(),
         'c_plus': shape.tolist(), 'c_minus': (-shape).tolist(),
         'note': 'u=+1 (c_plus): bins 6,7 pleasant, bins 0,1 unpleasant'}
    with open(os.path.join(OUT, 'cand_shape.json'), 'w') as f:
        json.dump(r, f, indent=1)
    print(json.dumps(r, indent=1))


if __name__ == '__main__':
    main()
