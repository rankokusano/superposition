"""
v11 stage-2: late-5 summary of t1t3_s2_v11.py outputs (docs/v11_experiment_log.md Sec.7.51).
Prints / writes mean [min, max] over the late-5 for every run, plus the data-only P_A2 values.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from t1t3_s2_v11 import OUT3  # noqa

EPS = [160, 170, 180, 190, 200]


def flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f'{pre}{k}/'))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                out[f'{pre}{k}[{i}]'] = x
        elif isinstance(v, (int, float)):
            out[f'{pre}{k}'] = v
    return out


def main():
    res = {}
    for st in ('green', 'pref'):
        for s in (0, 2):
            for c in ('random', 'mixed'):
                fs = [flat(json.load(open(os.path.join(OUT3, f'{st}_{c}_s{s}_ep{e}.json')))) for e in EPS]
                res[f'{st}_{c}_s{s}'] = {k: [float(np.mean([f[k] for f in fs])), float(min(f[k] for f in fs)),
                                             float(max(f[k] for f in fs))] for k in fs[0]}
    res['d6_PA2'] = flat(json.load(open(os.path.join(OUT3, 'd6_PA2.json'))))
    json.dump(res, open(os.path.join(OUT3, 'aggregate_t1t3.json'), 'w'), indent=1)
    keys = [k for k in res['green_random_s0'] if 'bin_mean' not in k]
    for k in keys:
        print(f'{k:42s}', ' '.join(f'{st[0]}{c[0]}{s}:{res[f"{st}_{c}_s{s}"][k][0]:+.3f}' for st in ('green', 'pref') for s in (0, 2) for c in ('random', 'mixed')))


if __name__ == '__main__':
    main()
