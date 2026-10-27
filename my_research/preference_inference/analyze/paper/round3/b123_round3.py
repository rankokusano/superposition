"""
Paper round 3, work 4 B1-B3 (paper/check/paper_round3_log.md; NOT pre-registered, saved values only, CPU).
B1: true-pleasure Delta (true - zero, other-side feature-prediction error) per condition x A-2 rule probability x pair x set,
    from data/result/paper_extra/A_cross/summary.json (L1-stage ep200; the other-side error is the same at every L1
    checkpoint, so the late-5 mean equals this value), with the paired episode bootstrap CI saved there; across 6 pairs
    the mean and a t 95% CI (df 5).
B2: self-position tracking (G1, motion-visible steps, common test) late-5 range for the sign-randomised condition
    (C_signrand/summary.json), the no-sharing condition (v13/ns/compare.json) and 80% in the same pairs.
B3: G5b sign-randomised - no experience per pair (C_signrand/summary.json 'G5b_vsnone'): late-5 points, CI > 0 count.
Writes paper/check/round3_b123.json.
"""

import json
import os

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
R = os.path.join(_PI, 'data', 'result')
OUT = os.path.join(_PI, 'paper', 'check', 'round3_b123.json')


def tci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), [float(v.mean() - h), float(v.mean() + h)]


def main():
    out = {'definition': 'paper_round3_log.md (work 4 B1-B3, not pre-registered)'}
    a = json.load(open(os.path.join(R, 'paper_extra', 'A_cross', 'summary.json')))['per_pair']
    b1 = {}
    for st in ('green', 'pref'):
        for c in ('random', 'mixed50', 'mixed80'):
            for p in ('0.5', '0.8'):
                rows = {}
                for s in range(6):
                    x = a[f'{st}_s{s}']['Delta'][f'{c}_{p}']
                    rows[f's{s}'] = {'mean': x['mean'], 'ci95': x['ci95'],
                                     'ci_excludes_0': bool(x['ci95'][0] > 0 or x['ci95'][1] < 0)}
                m, ci = tci([r['mean'] for r in rows.values()])
                b1[f'{st}|{c}|{p}'] = {'pairs': rows, 'across6_mean': m, 'across6_ci95': ci}
    out['B1'] = b1
    c = json.load(open(os.path.join(R, 'paper_extra', 'C_signrand', 'summary.json')))
    ns = json.load(open(os.path.join(R, 'v13', 'ns', 'compare.json')))
    b2 = {}
    for s in range(3):
        v = c['signrand']['per_pair'][f's{s}']['values']
        nsv = [ns[f's{s}_ep{e}']['scalars']['ns']['mixed']['G1_visible'] for e in (160, 170, 180, 190, 200)]
        v13 = [ns[f's{s}_ep{e}']['scalars']['v13']['mixed']['G1_visible'] for e in (160, 170, 180, 190, 200)]
        b2[f's{s}'] = {'signrand': v['signrand']['G1_visible'], 'pzero': c['pzero']['per_pair'][f's{s}']['values']['pzero']['G1_visible'],
                       '80': v['mixed80']['G1_visible'], 'ns': nsv, '80_from_ns_compare': v13,
                       'none': v['random']['G1_visible']}
    out['B2'] = b2
    b3 = {}
    for s in range(3):
        pts = c['signrand']['per_pair'][f's{s}']['G5b_vsnone']['points']
        b3[f's{s}'] = {'points': pts, 'mean_late5': float(np.mean([p['mean'] for p in pts])),
                       'n_ci_low_above0': int(sum(p['ci95'][0] > 0 for p in pts)),
                       'n_ci_high_below0': int(sum(p['ci95'][1] < 0 for p in pts))}
    out['B3'] = b3
    json.dump(out, open(OUT, 'w'), indent=1)
    for k, r in b1.items():
        print('B1', k, ' '.join(f"{p['mean']:+.3f}[{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}]{'' if p['ci_excludes_0'] else '*'}"
                                for p in r['pairs'].values()),
              f"| across {r['across6_mean']:+.3f} [{r['across6_ci95'][0]:+.3f}, {r['across6_ci95'][1]:+.3f}]")
    for s, r in b2.items():
        print('B2', s, {k: (round(min(v), 3), round(max(v), 3)) for k, v in r.items()})
    for s, r in b3.items():
        print('B3', s, round(r['mean_late5'], 3), [round(p['mean'], 3) for p in r['points']], 'CI>0', r['n_ci_low_above0'],
              'CI<0', r['n_ci_high_below0'])


if __name__ == '__main__':
    main()
