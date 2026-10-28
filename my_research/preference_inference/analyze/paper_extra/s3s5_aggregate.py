"""
Paper round 6 (docs/paper_extra_log.md Sec.34.3, pre-registered in bb6de41 + addendum b12683d): sign-randomised
condition, pairs 4-6 (s3-s5) and the 6-pair analysis. CPU, saved per-episode values only.
  per pair 4-6 : V1 (G5b, sign-following green / pref), V3 (d_P), V2 (true-pleasure Delta), P1 and the signrand - none
                 descriptions exactly as c_aggregate.py (Sec.4.4); B4-V1 exactly as b4_aggregate.py (Sec.29.4).
                 Pairs 1-3 are recomputed with the same functions only to check that they equal the saved summaries
                 (C_signrand/summary.json, B4/summary.json); their judgements are not redone.
  6 pairs      : per pair value = mean over the 5 late points of each model's nanmean (main) or of the episode-paired
                 difference mean (eval_g5_v12.summ; description); signrand - 80% over pairs 1-6 by
                 across_pairs_v13.test (t CI df 5, two-sided sign test); main 1 holds if the main-form CI upper < 0
                 for both d_P and G5b (common test). Pairs 1-3 only (df 2) alongside.
  exclusion    : outside the pre-registration (Sec.34.3.4): orig / a (all conditions of the pair) / a3 (none, 80%,
                 signrand) / b1 (unreadable -> 0 in both) / b2 (unreadable -> 0 in 80% only), G5b and sign-following.
  checks       : rep bit-identity (trial, G5, B4 G5b), g3flip hold / zero of 80% and none = saved G3, B4 common-test
                 tracking = saved G1, d_P NaN pattern identical across models, checkpoints saved.
-> data/result/paper_extra/C_signrand/s3s5/summary.json (+ tables printed as markdown)
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_PI, 'analyze', 'v13')):
    sys.path.insert(0, p)
import c_aggregate as CA  # noqa
import b4_aggregate as BA  # noqa
import r4_exclusion as R4  # noqa
from across_pairs_v13 import test as ttest  # noqa

RES = CA.RES
E = CA.E
EPS = CA.EPS
OUT = os.path.join(RES, 'paper_extra', 'C_signrand', 's3s5')
SEEDS = range(6)
NEW = (3, 4, 5)
summ = CA.summ


def q_common(s):
    """quantity -> fn(cond, e) on the common evaluation data (c_aggregate names: signrand, mixed80, random)."""
    return {'dP': lambda c, e: CA.g3(c, s, e, 'dP'),
            'G5b': lambda c, e: CA.g5(c, s, e, 'G5b'),
            'signfollow_green': lambda c, e: CA.sf(c, s, e, 'green'),
            'signfollow_pref': lambda c, e: CA.sf(c, s, e, 'pref'),
            'G3hold': lambda c, e: CA.g3(c, s, e, 'hold')}


def q_b4(s):
    return {'B4_G5b': lambda c, e: BA.g5b(c, s, e),
            'B4_signfollow_green': lambda c, e: BA.sf(c, s, e, 'green'),
            'B4_signfollow_pref': lambda c, e: BA.sf(c, s, e, 'pref')}


def model_mean(fn, c):
    return float(np.mean([np.nanmean(fn(c, e)) for e in EPS]))


def paired_mean(fn, a, b):
    return float(np.mean([summ(fn(a, e) - fn(b, e))['mean'] for e in EPS]))


def per_pair(s):
    """c_aggregate.main's per-pair block for X = signrand (pairs 4-6 judgements; pairs 1-3 for the check)."""
    P, X = {}, 'signrand'
    Q = q_common(s)
    for ref, nm in (('mixed80', 'vs80'), ('random', 'vsnone')):
        sign = -1 if ref == 'mixed80' else 1
        P[f'G5b_{nm}'] = CA.pointwise(lambda c, e: Q['G5b'](c, e), X, ref, sign)
        for k in ('green', 'pref'):
            P[f'signfollow_{k}_{nm}'] = CA.pointwise(Q[f'signfollow_{k}'], X, ref, sign)
        P[f'dP_{nm}'] = CA.pointwise(Q['dP'], X, ref, sign)
        P[f'G3hold_{nm}'] = CA.pointwise(Q['G3hold'], X, ref, sign)
        for k in ('green', 'pref'):
            b = CA.boot(CA.delta(X, s, k) - CA.delta(ref, s, k))
            P[f'delta_{k}_{nm}'] = {'diff': b, 'ok': CA.neg(b) if sign < 0 else CA.pos(b)}
    B = q_b4(s)
    for q, fn in B.items():
        P[f'{q}_vs80'] = BA.pointwise(fn, X, 'mixed80', -1)
        P[f'{q}_vsnone'] = BA.pointwise(fn, X, 'random', 1)
        P[f'{q}_80_vsnone'] = BA.pointwise(fn, 'mixed80', 'random', 1)
    return P


def values(s):
    """late-5 values of signrand / mixed80 / random (common test and B4)."""
    lv = {}
    for c in ('signrand', 'mixed80', 'random'):
        d = {q: [float(np.nanmean(fn(c, e))) for e in EPS] for q, fn in {**q_common(s), **q_b4(s)}.items()}
        for q in ('hold', 'zero', 'flip'):
            d[f'G3_{q}'] = [float(np.nanmean(CA.g3(c, s, e, q))) for e in EPS]
        d['g_rest_holdflip'] = [float(np.nanmean((CA.g3(c, s, e, 'hold') + CA.g3(c, s, e, 'flip')) / 2)) for e in EPS]
        js = [CA.trial_json(c, s, e) for e in EPS]
        d['G1_visible'] = [j['tracking']['p_next']['visible'] for j in js]
        d['self_err_visible'] = [j['err_mean_visible'] for j in js]
        d['self_err_hidden'] = [j['err_mean_hidden'] for j in js]
        d['delta_green'] = float(CA.delta(c, s, 'green').mean())
        d['delta_pref'] = float(CA.delta(c, s, 'pref').mean())
        if c != 'random':
            d['BA_L1_l1late5'] = [CA.inv(c, s, e, 'L1') for e in EPS]
            d['BA_FP_mselate5'] = [CA.inv(c, s, e, 'FP') for e in EPS]
        tr = [json.load(open(os.path.join(BA.B4, 'track', f'{c}_s{s}_ep{e}.json'))) for e in EPS]
        d['B4_track_visible'] = [t['p_next']['visible'] for t in tr]
        d['B4_track_hidden'] = [t['p_next']['hidden'] for t in tr]
        # readable fractions (Sec.30.2 definitions): G5b episode finite; sign-following freeze and flipped g finite
        d['read_G5b'] = [float(np.isfinite(CA.g5(c, s, e, 'G5b')).mean()) for e in EPS]
        for k in ('green', 'pref'):
            d[f'read_{k}'] = [float(np.isfinite(CA.sf(c, s, e, k)).mean()) for e in EPS]
            d[f'B4_read_{k}'] = [float(np.isfinite(BA.sf(c, s, e, k)).mean()) for e in EPS]
        d['B4_read_G5b'] = [float(np.isfinite(BA.g5b(c, s, e)).mean()) for e in EPS]
        d['read_dP'] = [float(np.isfinite(CA.g3(c, s, e, 'dP')).mean()) for e in EPS]
        lv[c] = d
    return lv


def checks(s):
    ck = {}
    a = np.load(os.path.join(E, 'g5', 'signrand', f's{s}', 'g5_per_ep200.npz'))
    b = np.load(os.path.join(E, 'g5_rep', 'signrand', f's{s}', 'g5_per_ep200.npz'))
    ck['g5_rep'] = all(np.array_equal(a[q], b[q], equal_nan=True) for q in a.files)
    ck['trial_rep'] = CA.trial_json('signrand', s, 200) == json.load(open(os.path.join(E, 's1eval', 'signrand', f's{s}', 'eval_signrand_ep200_rep.json')))
    for c, key in (('mixed80', 'mixed'), ('random', 'random')):
        ck[f'g3flip_{c}_equals_saved'] = all(
            np.array_equal(CA.g3(c, s, e, q), np.load(os.path.join(RES, 'v13', 'main', 's1eval', f's{s}', f'per_ep{e}.npz'))[f'{key}__freeze_K15_{q}'], equal_nan=True)
            for e in EPS for q in ('hold', 'zero'))
    ck['B4_G5b_rep'] = all(np.array_equal(BA.g5b(c, s, e), BA.g5full(c, s, e, 'G5b'), equal_nan=True)
                           for c in ('signrand', 'mixed80', 'random') for e in EPS)
    tc = {}
    for c in ('signrand', 'mixed80', 'random'):
        f = os.path.join(BA.B4, 'track_common', f'{c}_s{s}_ep200.json')
        tc[c] = json.load(open(f))['p_next']['visible'] == CA.trial_json(c, s, 200)['tracking']['p_next']['visible']
    ck['B4_track_common_equals_saved_G1'] = tc
    ck['dP_nan_pattern_same'] = all(
        np.array_equal(np.isnan(CA.g3('signrand', s, e, q)), np.isnan(CA.g3(c, s, e, q)))
        for c in ('mixed80', 'random') for e in EPS for q in ('hold', 'flip'))
    ck['ckpts'] = {'mse': len(os.listdir(os.path.join(RES, 'paper_extra', 'C_signrand', 's1_signrand_mse', str(s), 'model'))),
                   'l1': len(os.listdir(os.path.join(RES, 'paper_extra', 'C_signrand', f's1_signrand_l1_s{s}', str(s), 'model')))}
    return ck


# ---- exclusion variants (outside the pre-registration) -------------------------------------------------------------
R4M = {'signrand': 'signrand', 'mixed80': '80', 'random': 'none'}
POOL_A = {s: (R4.ALL13 if s <= 2 else ('none', '50', '80', 'signrand')) for s in SEEDS}
POOL_A3 = ('none', '80', 'signrand')


def variant_arrays(s, e, q, variant):
    """signrand and 80% per-episode arrays (common test; q in G5b / green / pref) under an exclusion variant."""
    X = {m: np.array(R4.val(m, s, e, q), float) for m in ('signrand', '80')}
    if variant in ('a', 'a3'):
        pool = POOL_A[s] if variant == 'a' else POOL_A3
        ok = np.all([np.isfinite(R4.val(m, s, e, q)) for m in pool], 0)
        for m in X:
            X[m][~ok] = np.nan
    elif variant == 'b1':
        for m in X:
            X[m][np.isnan(X[m])] = 0.0
    elif variant == 'b2':
        X['80'][np.isnan(X['80'])] = 0.0
    return X


def exclusion():
    out = {}
    for q, nm in (('G5b', 'G5b'), ('green', 'signfollow_green'), ('pref', 'signfollow_pref')):
        for v in ('orig', 'a', 'a3', 'b1', 'b2'):
            pp, dm, dp = {}, [], []
            for s in SEEDS:
                Xs = [variant_arrays(s, e, q, v) for e in EPS]
                pts = [summ(X['signrand'] - X['80']) for X in Xs]
                ok = [p['mean'] < 0 and p['ci95'][1] < 0 for p in pts]
                pp[f's{s}'] = {'n_ok': int(sum(ok)), 'pair_ok': sum(ok) >= 3}
                dm.append(float(np.mean([np.nanmean(X['signrand']) - np.nanmean(X['80']) for X in Xs])))
                dp.append(float(np.mean([p['mean'] for p in pts])))
            out[f'{nm}|{v}'] = {'per_pair': pp, 'pairs_ok_new3': sum(pp[f's{s}']['pair_ok'] for s in NEW),
                                'pairs_ok_all6': sum(p['pair_ok'] for p in pp.values()),
                                't_means': ttest(dm), 't_paired': ttest(dp), 'diff_means': dm, 'diff_paired': dp}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    C = json.load(open(os.path.join(RES, 'paper_extra', 'C_signrand', 'summary.json')))['signrand']
    B = json.load(open(os.path.join(RES, 'paper_extra', 'B4', 'summary.json')))
    R = {'per_pair': {}, 'values': {}, 'checks': {}}
    for s in SEEDS:
        R['per_pair'][f's{s}'] = per_pair(s)
        R['values'][f's{s}'] = values(s)
        if s in NEW:
            R['checks'][f's{s}'] = checks(s)
    # pairs 1-3: recomputed values must equal the saved summaries (judgements are taken from them, not redone)
    same = True
    for s in range(3):
        P, Cp, Bp = R['per_pair'][f's{s}'], C['per_pair'][f's{s}'], B['per_pair'][f's{s}']
        for k in ('G5b', 'signfollow_green', 'signfollow_pref', 'dP', 'G3hold'):
            for nm in ('vs80', 'vsnone'):
                same &= P[f'{k}_{nm}'] == Cp[f'{k}_{nm}']
        for k in ('green', 'pref'):
            for nm in ('vs80', 'vsnone'):
                same &= P[f'delta_{k}_{nm}'] == Cp[f'delta_{k}_{nm}']
        for q in ('G5b', 'signfollow_green', 'signfollow_pref'):
            same &= P[f'B4_{q}_vs80'] == Bp[f'{q}_sr_vs80'] and P[f'B4_{q}_vsnone'] == Bp[f'{q}_sr_vsnone'] \
                and P[f'B4_{q}_80_vsnone'] == Bp[f'{q}_80_vsnone']
    R['pairs123_equal_saved_summaries'] = same
    # per-pair verdicts, pairs 4-6 as Sec.4.4 (2 of 3) and counts over 6
    J = {}
    for k in ('G5b', 'signfollow_green', 'signfollow_pref', 'dP', 'G3hold', 'B4_G5b', 'B4_signfollow_green', 'B4_signfollow_pref'):
        for nm in ('vs80', 'vsnone'):
            J[f'{k}_{nm}'] = {'new3': sum(R['per_pair'][f's{s}'][f'{k}_{nm}']['pair_ok'] for s in NEW),
                              'all6': sum(R['per_pair'][f's{s}'][f'{k}_{nm}']['pair_ok'] for s in SEEDS),
                              'points': [R['per_pair'][f's{s}'][f'{k}_{nm}']['n_ok'] for s in SEEDS]}
    for k in ('green', 'pref'):
        for nm in ('vs80', 'vsnone'):
            J[f'delta_{k}_{nm}'] = {'new3': sum(R['per_pair'][f's{s}'][f'delta_{k}_{nm}']['ok'] for s in NEW),
                                    'all6': sum(R['per_pair'][f's{s}'][f'delta_{k}_{nm}']['ok'] for s in SEEDS)}
    v2 = [J[f'delta_{k}_vs80']['new3'] >= 2 for k in ('green', 'pref')]
    R['judge_counts'] = J
    R['verdict_new3'] = {'V1_G5b': J['G5b_vs80']['new3'] >= 2, 'V1_signfollow_green': J['signfollow_green_vs80']['new3'] >= 2,
                         'V1_signfollow_pref': J['signfollow_pref_vs80']['new3'] >= 2, 'V3': J['dP_vs80']['new3'] >= 2,
                         'V2': 'holds' if all(v2) else ('partly' if any(v2) else 'no'),
                         'P1_delta': {k: J[f'delta_{k}_vsnone']['new3'] >= 2 and not v for k, v in zip(('green', 'pref'), v2)},
                         'P1_G3hold': J['G3hold_vsnone']['new3'] >= 2,
                         **{f'B4V1_{q}': J[f'B4_{q}_vs80']['new3'] >= 2 for q in ('G5b', 'signfollow_green', 'signfollow_pref')}}
    # 6-pair t (main: difference of model means; description: episode-paired)
    T = {}
    for q in ('dP', 'G5b', 'signfollow_green', 'signfollow_pref', 'B4_G5b', 'B4_signfollow_green', 'B4_signfollow_pref', 'G3hold'):
        dm, dp = [], []
        for s in SEEDS:
            fn = {**q_common(s), **q_b4(s)}[q]
            dm.append(model_mean(fn, 'signrand') - model_mean(fn, 'mixed80'))
            dp.append(paired_mean(fn, 'signrand', 'mixed80'))
        T[q] = {'diff_means': dm, 'diff_paired': dp, 'all6_means': ttest(dm), 'all6_paired': ttest(dp),
                'main3_means': ttest(dm[:3]), 'main3_paired': ttest(dp[:3]),
                'n_neg_means': int(sum(x < 0 for x in dm)), 'n_neg_paired': int(sum(x < 0 for x in dp))}
        up_m, up_p = T[q]['all6_means']['ci95'][1] < 0, T[q]['all6_paired']['ci95'][1] < 0
        T[q]['forms_disagree'] = up_m != up_p or [x < 0 for x in dm] != [x < 0 for x in dp]
    R['across6'] = T
    R['main1_6pairs_main_form'] = T['dP']['all6_means']['ci95'][1] < 0 and T['G5b']['all6_means']['ci95'][1] < 0
    R['main1_6pairs_paired_form'] = T['dP']['all6_paired']['ci95'][1] < 0 and T['G5b']['all6_paired']['ci95'][1] < 0
    R['exclusion'] = exclusion()
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(R, f, indent=1)
    print('pairs 1-3 equal saved summaries:', same)
    print('checks', json.dumps(R['checks']))
    print('verdict pairs 4-6', json.dumps(R['verdict_new3']))
    print('judge counts', json.dumps({k: (v['new3'], v['all6'], v.get('points')) for k, v in J.items()}))
    for q, t in T.items():
        f = lambda r: f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] p={r['sign_p']:.3f}"  # noqa
        print(f'{q:20s} means {f(t["all6_means"])} neg {t["n_neg_means"]}/6 | paired {f(t["all6_paired"])} neg {t["n_neg_paired"]}/6 | disagree {t["forms_disagree"]}')
    print('main 1 (6 pairs): main form', R['main1_6pairs_main_form'], '| paired form', R['main1_6pairs_paired_form'])


if __name__ == '__main__':
    main()
