"""
v14-C gate judgement (docs/v14_experiment_log.md Sec.31) from gates_v14c.py outputs. Rules as v12
(aggregate_s1_v12 / aggregate_g5_v12), applied per subset of episodes:
  G1: visible-step tracking R^2 >= 0.80 for both models;  G2: M1c (exp hidden c_pred < 0, paired CI upper < 0),
  M1d (exp hidden c_cf > 0, paired CI lower > 0), colour (exp hidden S_L - S_D on swap episodes, CI lower > 0);
  G3: exp g (K15 hold) CI lower > 0 and paired CI lower > 0;  (mse late-5)
  G5a: in both sets (1 green, 2 pref), exp g_other CI lower > 0 and paired CI lower > 0 (A-1 Red = main, Green
  alongside);  G5b: exp CI lower > 0 and paired CI lower > 0.  (l1 late-5)
Subsets: all episodes (main), A-1 liked Red / Green (G1-G3, G5b); A-2 liked Red / Green (G5a, pref set only).
A pair passes with >= 3 of 5 checkpoints; the gate with >= 2 of 3 pairs (main: partner v12noexp, all episodes,
G5a with A-1 Red). Paired values: per-episode experienced - partner, 1000 bootstrap (v12 summ).
Usage: python analyze/v14/aggregate_gates_v14c.py --partner v12noexp --seeds 0 1 2
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
from eval_trial_v12 import summ  # noqa

BASE = os.path.join(_PI, 'data', 'result', 'v14', 'gates')
EPOCHS = (160, 170, 180, 190, 200)


def ld(s, mode, c, e):
    f = os.path.join(BASE, f's{s}', f'{mode}_{c}_ep{e}')
    return np.load(f + '.npz'), json.load(open(f + '.json'))


def S(v):
    v = np.asarray(v, float)
    return summ(v) if np.isfinite(v).sum() > 1 else {'mean': float('nan'), 'ci95': [float('nan')] * 2, 'n': 0}


def s1_point(ze, zp, je, jp, mask, sub):
    r = {}
    m1c, m1cp = S(ze['hidden_c_pred'][mask]), S((ze['hidden_c_pred'] - zp['hidden_c_pred'])[mask])
    m1d, m1dp = S(ze['hidden_c_cf'][mask]), S((ze['hidden_c_cf'] - zp['hidden_c_cf'])[mask])
    col = S(ze['hidden_colour_swap'][mask])
    g3, g3p = S(ze['g_K15_hold'][mask]), S((ze['g_K15_hold'] - zp['g_K15_hold'])[mask])
    r['M1c'] = bool(m1c['mean'] < 0 and m1cp['ci95'][1] < 0)
    r['M1d'] = bool(m1d['mean'] > 0 and m1dp['ci95'][0] > 0)
    r['colour'] = bool(col['ci95'][0] > 0)
    r['G3'] = bool(g3['ci95'][0] > 0 and g3p['ci95'][0] > 0)
    r['G1'] = bool(min(je['tracking'][sub]['visible'], jp['tracking'][sub]['visible']) >= 0.80)
    r['values'] = {'M1c': m1c, 'M1c_paired': m1cp, 'M1d': m1d, 'M1d_paired': m1dp, 'colour_swap': col,
                   'colour_all': S(ze['hidden_colour_all'][mask]), 'G3': g3, 'G3_paired': g3p,
                   'G3_partner': S(zp['g_K15_hold'][mask]),
                   'track_visible': [je['tracking'][sub]['visible'], jp['tracking'][sub]['visible']]}
    return r


def g5_point(ze, zp, a1, a2mask=None):
    r, vals = {}, {}
    ok = []
    for k in ('green', 'pref'):
        key = f'G5a_a1{a1}_{k}_K15'
        e_, p_ = ze[key], ze[key] - zp[key]
        if a2mask is not None:
            if k != 'pref':
                continue
            e_, p_ = e_[a2mask], p_[a2mask]
        ve, vp = S(e_), S(p_)
        vals[k] = {'exp': ve, 'paired': vp, 'partner': S(zp[key] if a2mask is None else zp[key][a2mask])}
        ok.append(ve['ci95'][0] > 0 and vp['ci95'][0] > 0)
    r['G5a'] = bool(all(ok))
    r['values'] = vals
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partner', default='v12noexp')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = ap.parse_args()
    res = {'partner': a.partner, 'per_seed': {}}
    for s in a.seeds:
        rs = {}
        for e in EPOCHS:
            ze, je = ld(s, 's1', 'exp', e)
            zp, jp = ld(s, 's1', a.partner, e)
            lk = ze['liked']
            for sub, mk in (('all', np.ones(len(lk), bool)), ('red', lk == 1), ('green', lk == 0)):
                rs.setdefault(f's1_{sub}', []).append(s1_point(ze, zp, je, jp, mk, sub))
            ge, _ = ld(s, 'g5', 'exp', e)
            gp, _ = ld(s, 'g5', a.partner, e)
            for a1 in ('red', 'green'):
                rs.setdefault(f'g5a_a1{a1}', []).append(g5_point(ge, gp, a1))
                a2 = ge['a2_liked_pref']
                for nm, mk in (('a2red', a2 == 1), ('a2green', a2 == 0)):
                    rs.setdefault(f'g5a_a1{a1}_{nm}_prefonly', []).append(g5_point(ge, gp, a1, mk))
            lk5 = ge['liked']
            for sub, mk in (('all', np.ones(len(lk5), bool)), ('red', lk5 == 1), ('green', lk5 == 0)):
                be, bp = S(ge['G5b'][mk]), S((ge['G5b'] - gp['G5b'])[mk])
                rs.setdefault(f'g5b_{sub}', []).append({'G5b': bool(be['ci95'][0] > 0 and bp['ci95'][0] > 0),
                                                       'values': {'exp': be, 'paired': bp}})
        counts = {}
        for key, rows in rs.items():
            for crit in [k for k in rows[0] if k != 'values']:
                counts[f'{key}|{crit}'] = sum(r[crit] for r in rows)
        res['per_seed'][f's{s}'] = {'rows': rs, 'counts': counts}
    need = {1: 1, 2: 2, 3: 2, 4: 3}[len(a.seeds)]
    keys = sorted(res['per_seed'][f's{a.seeds[0]}']['counts'])
    res['gate'] = {k: {'pairs_passed': [s for s in res['per_seed'] if res['per_seed'][s]['counts'][k] >= 3],
                       'counts': [res['per_seed'][s]['counts'][k] for s in res['per_seed']]} for k in keys}
    for k in keys:
        res['gate'][k]['pass'] = len(res['gate'][k]['pairs_passed']) >= need
    main_keys = ['s1_all|G1', 's1_all|M1c', 's1_all|M1d', 's1_all|colour', 's1_all|G3', 'g5a_a1red|G5a', 'g5b_all|G5b']
    res['main'] = {k: res['gate'][k] for k in main_keys}
    res['main']['G2'] = all(res['gate'][k]['pass'] for k in ('s1_all|M1c', 's1_all|M1d', 's1_all|colour'))
    res['main']['to_stage2'] = res['main']['G2'] and res['gate']['s1_all|G3']['pass'] and \
        res['gate']['g5a_a1red|G5a']['pass'] and res['gate']['g5b_all|G5b']['pass']
    json.dump(res, open(os.path.join(BASE, f'aggregate_gates_{a.partner}.json'), 'w'), indent=1)
    for k in keys:
        print(f"{k:40s} counts {res['gate'][k]['counts']}  pass {res['gate'][k]['pass']}")
    print('G2', res['main']['G2'], 'to_stage2', res['main']['to_stage2'])


if __name__ == '__main__':
    main()
