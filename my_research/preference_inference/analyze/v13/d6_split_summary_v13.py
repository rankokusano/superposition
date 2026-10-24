"""
Summary of analyze/v13/d6_split_v13.py (docs/v13_experiment_log.md Sec.24; post-hoc). For each pair, set and
model: mean other-side feature-prediction error over frames where A-2's step followed its preference
(a2_source 1), did not (0 or 2) and slept (3); per episode first, then the mean over episodes with a bootstrap CI.
Differences: D6 - zero (same model), and v13 - v12 (mixed80 - mixed50, same frames). Writes summary.json and
prints a markdown table.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES, boot_ci  # noqa

ROOT = os.path.join(RES, 'v13', 'post', 'd6_split')
CLS = {'follow': (1,), 'notfollow': (0, 2), 'sleep': (3,)}


def ep_mean(e, src, codes):
    m = np.isin(src, codes)
    with np.errstate(invalid='ignore'):
        return np.where(m.any(1), (e * m).sum(1) / m.sum(1), np.nan)


def summ(v):
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def main():
    out = {}
    for s in (0, 1, 2):
        d = {c: np.load(os.path.join(ROOT, f's{s}', f'{c}.npz')) for c in ('random', 'mixed50', 'mixed80')}
        for k in ('green', 'pref'):
            src = d['random'][f'{k}_a2_source']
            for cl, codes in CLS.items():
                v = {c: {var: ep_mean(d[c][f'{k}_{var}'], src, codes) for var in ('d6', 'zero')} for c in d}
                r = {f'{c}_{var}': summ(v[c][var]) for c in d for var in ('d6', 'zero')}
                for c in d:
                    r[f'{c}_d6_minus_zero'] = summ(v[c]['d6'] - v[c]['zero'])
                r['v13_minus_v12_d6'] = summ(v['mixed80']['d6'] - v['mixed50']['d6'])
                r['frac_frames'] = float(np.isin(src, codes).mean())
                out[f's{s}_{k}_{cl}'] = r
    json.dump(out, open(os.path.join(ROOT, 'summary.json'), 'w'), indent=1)

    def rng(key, q):
        v = [out[f's{s}_{key}'][q]['mean'] for s in (0, 1, 2)]
        return f'{min(v):.3f}〜{max(v):.3f}'
    print('| 集合 | 一歩 | 割合 | 経験なし D6 | 50% D6 | 80% D6 | 80% 零 | 80% D6 − 零 | v13 − v12（D6） |')
    print('|---|---|---|---|---|---|---|---|---|')
    for k in ('green', 'pref'):
        for cl in CLS:
            key = f'{k}_{cl}'
            fr = np.mean([out[f's{s}_{key}']['frac_frames'] for s in (0, 1, 2)])
            dd = [out[f's{s}_{key}']['v13_minus_v12_d6'] for s in (0, 1, 2)]
            cell = ' / '.join(f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" for x in dd)
            dz = [out[f's{s}_{key}']['mixed80_d6_minus_zero']['mean'] for s in (0, 1, 2)]
            print(f"| {'①' if k == 'green' else '②'} | {cl} | {fr:.2f} | {rng(key, 'random_d6')} | {rng(key, 'mixed50_d6')} | "
                  f"{rng(key, 'mixed80_d6')} | {rng(key, 'mixed80_zero')} | {min(dz):+.3f}〜{max(dz):+.3f} | {cell} |")


if __name__ == '__main__':
    main()
