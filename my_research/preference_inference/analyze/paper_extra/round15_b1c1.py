"""
Paper round 15, b1 and c1 (docs/paper_extra_log.md Sec.39.3; ALL outside the pre-registration, descriptive only).
CPU only, saved arrays only (no model is run).
  b1 size of the preference-shaped loss difference vs the reduction G achieved, other-side feature error (FP),
     v11_s2_pref test (300; 'pref', main) and v11_s2t_green test (300; 'green', secondary), pairs 1-4, 50% / none:
       U_pref = FP(opp) - FP(true) per episode (E2/runs, stage-I MSE stage; process-2 = -P_A2 / P_A2 held)
       D_G    = FP(zero) - FP(G's Q2) per episode (v12/stage2/t1t3 *_per.npz, stage-II G pref_<c>_s<s> on stage-I
                L1 ep200; FP does not change in the L1 stage, so its FP weights are those of MSE ep200)
     common window: steps t = 0..99 (T3 saves only the 0..99 episode mean). Main checkpoints: E2 at MSE ep200 (the
     point that shares the G base's FP weights), G late 5 (ep160-200) mean. Also: E2 late-5 mean, E2 frames 1-50, G ep200.
     Checks: per episode E2 ep200 FP(zero) vs T3 FP(zero) and E2 FP(true) vs T3 FP(d6 = P_A2), max |diff|.
     Ratio per pair = mean U_pref / mean D_G; also U_use / D_G (U_use = FP(zero) - FP(true)). 4-pair mean, t 95% (df 3).
  c1 U_use = L(zero) - L(true) for FP (E2/runs, MSE late 5, frames 1-50 = t 0..49; also 1-100) and U^pos_use =
     dev(zero) - dev(true) (E2/pos/runs, L1 late 5, readable frames 10-50, A-1 Red), with e2pos_aggregate.comparison
     unchanged (kind 'use'): p50 - none and p50 - sr50, 6 pairs, t 95% CI (df 5) of the pair-value difference (main
     form) and of the episode-paired difference; per pair: points with an episode-paired bootstrap CI > 0 (of 5).
     Main set: preference-mixed ('pref'); the other 3 sets descriptive.
-> data/result/paper_extra/Round15/{b1.json, c1.json}
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import e2pos_aggregate as P  # noqa  (E2 position aggregation; ttest, summ, RES)

RES = P.RES
OUT = os.path.join(RES, 'paper_extra', 'Round15')
E2R = os.path.join(RES, 'paper_extra', 'E2', 'runs')
T3 = os.path.join(RES, 'v12', 'stage2', 't1t3')
EPS = (160, 170, 180, 190, 200)
GC = {'p50': 'mixed', 'none': 'random'}


def e2(f, s, e, k):
    with np.load(os.path.join(E2R, f'{f}_mse_s{s}_ep{e}_{k}.npz')) as z:
        return {c: z[f'FP_{c}'].astype(np.float64) for c in ('true', 'opp', 'zero')}


def t3(c, s, e, k):
    with np.load(os.path.join(T3, f'pref_{GC[c]}_s{s}_ep{e}_per.npz')) as z:
        return {q: z[f'{k}_{q}_fp'].astype(np.float64) for q in ('gen', 'zero', 'd6')}


def tci(v):
    t = P.ttest(list(v))
    return {'mean': t['mean'], 'ci95': t['ci95'], 'per_pair': [float(x) for x in v]}


def b1():
    R = {'note': 'outside the pre-registration (Sec.39.3), descriptive', 'sets': {}}
    for k in ('pref', 'green'):
        Rk = {'check': {}, 'cond': {}}
        for c in ('p50', 'none'):
            rows = []
            for s in range(4):
                a = e2(c, s, 200, k)
                g = {e: t3(c, s, e, k) for e in EPS}
                Rk['check'][f'{c}_s{s}'] = {
                    'zero_maxabs': float(max(np.abs(a['zero'].mean(1) - g[e]['zero']).max() for e in EPS)),
                    'true_vs_d6_maxabs': float(max(np.abs(a['true'].mean(1) - g[e]['d6']).max() for e in EPS))}
                up = (a['opp'] - a['true']).mean(1).mean()
                uu = (a['zero'] - a['true']).mean(1).mean()
                dg = float(np.mean([(g[e]['zero'] - g[e]['gen']).mean() for e in EPS]))
                dg200 = float((g[200]['zero'] - g[200]['gen']).mean())
                late = [e2(c, s, e, k) for e in EPS]
                rows.append({'U_pref_ep200': float(up), 'U_use_ep200': float(uu), 'D_G_late5': dg, 'D_G_ep200': dg200,
                             'ratio_pref': float(up / dg), 'ratio_use': float(uu / dg),
                             'U_pref_late5_t0_99': float(np.mean([(z['opp'] - z['true']).mean() for z in late])),
                             'U_pref_late5_t0_49': float(np.mean([(z['opp'] - z['true'])[:, :50].mean() for z in late])),
                             'FP_zero': float(a['zero'].mean()), 'FP_true': float(a['true'].mean()),
                             'FP_gen_late5': float(np.mean([g[e]['gen'].mean() for e in EPS]))})
            Rk['cond'][c] = {'per_pair': rows, 'summary': {q: tci([r[q] for r in rows]) for q in rows[0]}}
        R['sets'][k] = Rk
    return R


def c1():
    R = {'note': 'outside the pre-registration (Sec.39.3), descriptive', 'cells': {}}
    for k in P.SETS:
        for q in ('FP|1-50', 'FP|1-100', 'pos'):
            cell = {}
            for nm, (a, b) in (('p50_none', ('p50', 'none')), ('p50_sr50', ('p50', 'sr50'))):
                r = P.comparison(a, b, k, q, kind='use')
                cell[nm] = {'t': r['t'], 't_paired': r['t_paired'], 'diffs': r['diffs'], 'diffs_paired': r['diffs_paired'],
                            'n_pos': r['n_pos'], 'pairs_ok': r['pairs_ok'], 'n_ok': [x['n_ok'] for x in r['per_pair']]}
            cell['alone'] = {f: P.ttest([P.pv(f, s, k, q, 'use') for s in P.SEEDS]) for f in ('none', 'p50', 'sr50')}
            R['cells'][f'{k}|{q}'] = cell
    return R


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rb = b1()
    rb['t_s'] = time.time() - t0
    json.dump(rb, open(os.path.join(OUT, 'b1.json'), 'w'), indent=1)
    print('b1', json.dumps({k: {c: {q: round(v['mean'], 4) for q, v in rb['sets'][k]['cond'][c]['summary'].items()}
                                for c in rb['sets'][k]['cond']} for k in rb['sets']}), flush=True)
    print('b1 checks', json.dumps(rb['sets']['pref']['check']), flush=True)
    rc = c1()
    rc['t_s'] = time.time() - t0
    json.dump(rc, open(os.path.join(OUT, 'c1.json'), 'w'), indent=1)
    for cell in ('pref|FP|1-50', 'pref|pos'):
        print('c1', cell, json.dumps({nm: (v['t']['mean'], v['t']['ci95'], v['t_paired']['ci95'])
                                      for nm, v in rc['cells'][cell].items() if nm != 'alone'}), flush=True)
    print('t', round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
