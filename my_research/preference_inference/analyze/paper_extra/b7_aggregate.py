"""
Paper round 5, work 1 (B7; docs/paper_extra_log.md Sec.31): checks, judgement and description from b7_dp.py outputs.
  checks   : hold / zero g bit-identical to the saved values (v14/gates/s<s>/s1_{exp,v12noexp,noexpv}_ep<e>.npz
             g_K15_hold / g_K15_zero; v12/main/s1eval/s<s>/per_ep<e>.npz mixed__freeze_K15_hold / _zero); ep200 twice.
  judgement: per mse late-5 point, pref d_P CI > 0 and (pref - nonesigned) paired CI > 0 (eval_trial_v11.boot_ci);
             3 of 5 points -> pair, 2 of 3 pairs -> holds.
  desc     : the same for Red- / Green-liked episodes; mixed50 d_P and mixed50 - none (existing g3flip random,
             sign +1) in the same form; late-5 means and pref / mixed50 ratio; noexpv; existing none d_P.
-> data/result/paper_extra/B7/summary.json
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
from eval_trial_v11 import boot_ci  # noqa

RES = os.path.join(_PI, 'data', 'result')
B7 = os.path.join(RES, 'paper_extra', 'B7')
EPS = (160, 170, 180, 190, 200)
SAVED = {'pref': 'exp', 'nonesigned': 'v12noexp', 'noexpv': 'noexpv'}


def ld(c, s, e, tag=''):
    return np.load(os.path.join(B7, f'{c}_s{s}_ep{e}{tag}.npz'))


def none_dp(s, e):
    return np.load(os.path.join(RES, 'paper_extra', 'C_signrand', 'eval', 'g3flip', f'random_s{s}_ep{e}.npz'))['dP']


def summ(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def pos(r):
    return r['mean'] > 0 and r['ci95'][0] > 0


def gate(xs, ys):
    pts = []
    for x, y in zip(xs, ys):
        r = {'exp': summ(x), 'diff': summ(x - y)}
        r['ok'] = pos(r['exp']) and pos(r['diff'])
        pts.append(r)
    n = sum(p['ok'] for p in pts)
    return {'points': pts, 'n_ok': n, 'pair_ok': n >= 3}


def main():
    out = {'checks': {}, 'per_pair': {}}
    for s in range(3):
        chk = {}
        for c, sv in SAVED.items():
            if c == 'noexpv' and s == 2:
                continue
            for e in EPS:
                a = ld(c, s, e)
                b = np.load(os.path.join(RES, 'v14', 'gates', f's{s}', f's1_{sv}_ep{e}.npz'))
                chk[f'{c}_ep{e}'] = all(np.array_equal(a[q], b[f'g_K15_{q}'], equal_nan=True) for q in ('hold', 'zero'))
            r = ld(c, s, 200, '_rep')
            chk[f'{c}_rep200'] = all(np.array_equal(ld(c, s, 200)[q], r[q], equal_nan=True) for q in ('hold', 'zero', 'flip'))
        for e in EPS:
            a = ld('mixed50', s, e)
            b = np.load(os.path.join(RES, 'v12', 'main', 's1eval', f's{s}', f'per_ep{e}.npz'))
            chk[f'mixed50_ep{e}'] = all(np.array_equal(a[q], b[f'mixed__freeze_K15_{q}'], equal_nan=True) for q in ('hold', 'zero'))
        r = ld('mixed50', s, 200, '_rep')
        chk['mixed50_rep200'] = all(np.array_equal(ld('mixed50', s, 200)[q], r[q], equal_nan=True) for q in ('hold', 'zero', 'flip'))
        out['checks'][f's{s}'] = chk

        P = {}
        liked = ld('pref', s, 200)['liked']
        dp = {c: [ld(c, s, e)['dP'] for e in EPS] for c in ('pref', 'nonesigned', 'mixed50')}
        nd = [none_dp(s, e) for e in EPS]
        P['pref_vs_nonesigned'] = gate(dp['pref'], dp['nonesigned'])
        for nm, mk in (('red', liked == 1), ('green', liked == 0)):
            P[f'pref_vs_nonesigned_{nm}liked'] = gate([x[mk] for x in dp['pref']], [y[mk] for y in dp['nonesigned']])
        P['mixed50_vs_none'] = gate(dp['mixed50'], nd)
        means = {c: [float(np.nanmean(x)) for x in v] for c, v in dp.items()}
        means['none_existing'] = [float(np.nanmean(x)) for x in nd]
        for nm, mk in (('red', liked == 1), ('green', liked == 0)):
            means[f'pref_{nm}liked'] = [float(np.nanmean(x[mk])) for x in dp['pref']]
            means[f'nonesigned_{nm}liked'] = [float(np.nanmean(x[mk])) for x in dp['nonesigned']]
        if s < 2:
            means['noexpv'] = [float(np.nanmean(ld('noexpv', s, e)['dP'])) for e in EPS]
        for c in ('pref', 'nonesigned', 'mixed50'):
            for q in ('hold', 'zero', 'flip'):
                means[f'{c}_{q}'] = [float(np.nanmean(ld(c, s, e)[q])) for e in EPS]
        P['late5_means'] = means
        P['ratio_pref_over_mixed50'] = float(np.mean(means['pref']) / np.mean(means['mixed50']))
        out['per_pair'][f's{s}'] = P
    J = {k: sum(out['per_pair'][f's{s}'][k]['pair_ok'] for s in range(3))
         for k in ('pref_vs_nonesigned', 'pref_vs_nonesigned_redliked', 'pref_vs_nonesigned_greenliked', 'mixed50_vs_none')}
    out['pairs_ok'] = J
    out['verdict_pref'] = J['pref_vs_nonesigned'] >= 2
    out['all_checks_ok'] = all(v for c in out['checks'].values() for v in c.values())
    with open(os.path.join(B7, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('checks ok', out['all_checks_ok'], 'pairs_ok', J, 'verdict', out['verdict_pref'])
    for s in range(3):
        P = out['per_pair'][f's{s}']
        print(f's{s}', {k: P[k]['n_ok'] for k in J}, {k: round(float(np.mean(v)), 3) for k, v in P['late5_means'].items()},
              'ratio', round(P['ratio_pref_over_mixed50'], 3))


if __name__ == '__main__':
    main()
