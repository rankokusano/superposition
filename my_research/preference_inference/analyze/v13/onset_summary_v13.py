"""
Summary of analyze/v13/onset_v13.py (docs/v13_experiment_log.md Sec.12.1; descriptive, gates unchanged).
Per pair and late-5 checkpoint: experienced 80% and 50% values, paired difference vs no experience, and
v13 - v12 (same episodes); 'stronger' counted as in Sec.3.4. Writes data/result/v13/main/onset/onset_summary.json
and prints a markdown table (late-5 ranges of the means; CI-based counts).
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES, boot_ci  # noqa

EPS = (160, 170, 180, 190, 200)
Q = ('G3_onset', 'G3_total', 'G5a_green_onset', 'G5a_green_total', 'G5a_pref_onset', 'G5a_pref_total')


def summ(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def ok(x):
    return x['mean'] > 0 and x['ci95'][0] > 0


def main():
    root = os.path.join(RES, 'v13', 'main', 'onset')
    out = {}
    for s in (0, 1, 2):
        for e in EPS:
            d = {c: np.load(os.path.join(root, f's{s}', f'onset_{c}_ep{e}.npz')) for c in ('random', 'mixed50', 'mixed80')}
            out[f's{s}_ep{e}'] = {q: {'random': summ(d['random'][q]), 'mixed50': summ(d['mixed50'][q]), 'mixed80': summ(d['mixed80'][q]),
                                      'paired50': summ(d['mixed50'][q] - d['random'][q]), 'paired80': summ(d['mixed80'][q] - d['random'][q]),
                                      'v13_minus_v12': summ(d['mixed80'][q] - d['mixed50'][q])} for q in Q}
    verdict = {}
    for q in Q:
        for k in ('mixed80', 'paired80', 'mixed50', 'paired50', 'v13_minus_v12'):
            pts = {f's{s}': sum(ok(out[f's{s}_ep{e}'][q][k]) for e in EPS) for s in (0, 1, 2)}
            verdict.setdefault(q, {})[k] = {'points': pts, 'pairs_ok': sum(v >= 3 for v in pts.values())}
    with open(os.path.join(root, 'onset_summary.json'), 'w') as f:
        json.dump({'per_ckpt': out, 'verdict': verdict}, f, indent=1)

    def rng(q, k):
        v = [out[f's{s}_ep{e}'][q][k]['mean'] for s in (0, 1, 2) for e in EPS]
        return f'{min(v):+.2f}〜{max(v):+.2f}'
    print('| 量 | 経験なし | 50% 経験あり | 50% 組の差 | 80% 経験あり | 80% 組の差 | v13 − v12 | v13 − v12 が正で CI が 0 を含まない点（s0/s1/s2） |')
    print('|---|---|---|---|---|---|---|---|')
    for q in Q:
        p = verdict[q]['v13_minus_v12']['points']
        print(f"| {q} | {rng(q, 'random')} | {rng(q, 'mixed50')} | {rng(q, 'paired50')} | {rng(q, 'mixed80')} | {rng(q, 'paired80')} | "
              f"{rng(q, 'v13_minus_v12')} | {p['s0']}/{p['s1']}/{p['s2']}（{verdict[q]['v13_minus_v12']['pairs_ok']} 組） |")
    print(json.dumps({q: {k: v['pairs_ok'] for k, v in verdict[q].items()} for q in Q}))


if __name__ == '__main__':
    main()
