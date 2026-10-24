"""
G5b by amount of experience within the same seed (docs/v13_experiment_log.md Sec.27): experienced 50% (v12) vs
80% (v13), 4 pairs (s0-s3), late-5 of the l1 stage. Per-episode values from eval_g5_v13 (v13: main/g5, v12 50%:
v12ref/g5, no experience: main/g5 'random'); v13 - v12 paired on the same episodes, CI eval_trial_v11.boot_ci.
Writes data/result/v13/main/g5b_dose.json and prints a markdown table.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES, boot_ci  # noqa

EPS = (160, 170, 180, 190, 200)


def summ(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def main():
    out = {}
    for s in (0, 1, 2, 3):
        for e in EPS:
            a = np.load(os.path.join(RES, 'v13', 'main', 'g5', f's{s}', f'g5_per_ep{e}.npz'))
            b = np.load(os.path.join(RES, 'v13', 'v12ref', 'g5', f's{s}', f'g5_per_ep{e}.npz'))
            m80, m50, r = a['mixed__G5b'], b['mixed__G5b'], a['random__G5b']
            out[f's{s}_ep{e}'] = {'none': summ(r), 'mixed50': summ(m50), 'mixed80': summ(m80),
                                  'v13_minus_v12': summ(m80 - m50), 'paired80': summ(m80 - r), 'paired50': summ(m50 - r)}
    json.dump(out, open(os.path.join(RES, 'v13', 'main', 'g5b_dose.json'), 'w'), indent=1)
    print('| 組 | 経験なし | 50%（v12） | 80%（v13） | 80% ÷ 50% | v13 − v12 が正で CI が 0 を含まない点 | 組の差 80% / 50%（正で CI が 0 を含まない点） |')
    print('|---|---|---|---|---|---|---|')
    for s in (0, 1, 2, 3):
        rr = [out[f's{s}_ep{e}'] for e in EPS]
        def rg(k):
            v = [x[k]['mean'] for x in rr]
            return f'{min(v):+.2f}〜{max(v):+.2f}'
        ratio = [x['mixed80']['mean'] / x['mixed50']['mean'] for x in rr]
        ok = sum(x['v13_minus_v12']['mean'] > 0 and x['v13_minus_v12']['ci95'][0] > 0 for x in rr)
        p80 = sum(x['paired80']['ci95'][0] > 0 for x in rr)
        p50 = sum(x['paired50']['ci95'][0] > 0 for x in rr)
        print(f"| s{s}{'（予備）' if s == 3 else ''} | {rg('none')} | {rg('mixed50')} | {rg('mixed80')} | {min(ratio):.1f}〜{max(ratio):.1f} 倍 | "
              f"{ok}/5（v13 − v12 {rg('v13_minus_v12')}） | {p80}/5 / {p50}/5 |")


if __name__ == '__main__':
    main()
