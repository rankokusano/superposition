"""
v11 stage-2: late-5 summary of t4t6_s2_v11.py (T4, T6, A-2 tracking) and m1d_other_v11.py
(M1(d)-other) outputs (docs/v11_experiment_log.md Sec.7.52-7.53). Mean over the late-5 per run;
the per-episode paired difference mixed - random (bootstrap 1000, seed 0) for the T4 score and
c_cf_other, per checkpoint.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from aggregate_t1t3_v11 import flat  # noqa
from t4t6_s2_v11 import OUT4  # noqa
from m1d_other_v11 import OUTM  # noqa

EPS = [160, 170, 180, 190, 200]


def boot(a, n=1000, seed=0):
    a = a[np.isfinite(a)]
    rng = np.random.default_rng(seed)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(a.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['t4t6', 'm1d_other'], required=True)
    args = ap.parse_args()
    root = OUT4 if args.which == 't4t6' else OUTM
    res = {}
    for st in ('green', 'pref'):
        for s in (0, 2):
            for c in ('random', 'mixed'):
                fs = [flat(json.load(open(os.path.join(root, f'{st}_{c}_s{s}_ep{e}.json')))) for e in EPS]
                res[f'{st}_{c}_s{s}'] = {k: float(np.nanmean([f[k] for f in fs])) for k in fs[0] if k != 'calib[0]' and k != 'calib[1]'}
            pk = [k for k in np.load(os.path.join(root, f'{st}_mixed_s{s}_ep200_per.npz')).files]
            res[f'{st}_s{s}_paired_mixed_minus_random'] = {
                k: [boot(np.load(os.path.join(root, f'{st}_mixed_s{s}_ep{e}_per.npz'))[k]
                         - np.load(os.path.join(root, f'{st}_random_s{s}_ep{e}_per.npz'))[k]) for e in EPS] for k in pk}
    json.dump(res, open(os.path.join(root, f'aggregate_{args.which}.json'), 'w'), indent=1)
    keys = list(res['green_random_s0'])
    for k in keys:
        print(f'{k:60s}', ' '.join(f'{st[0]}{c[0]}{s}:{res[f"{st}_{c}_s{s}"][k]:+.3f}' for st in ('green', 'pref') for s in (0, 2) for c in ('random', 'mixed')))
    for st in ('green', 'pref'):
        for s in (0, 2):
            for k, v in res[f'{st}_s{s}_paired_mixed_minus_random'].items():
                print(f'{st} s{s} {k} m-r:', ' '.join(f'{x[0]:+.3f}[{x[1][0]:+.3f},{x[1][1]:+.3f}]' for x in v))


if __name__ == '__main__':
    main()
