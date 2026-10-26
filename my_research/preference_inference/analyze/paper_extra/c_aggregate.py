"""
Paper extra C, judgement and description (docs/paper_extra_log.md Sec.4.4, 14.3; pre-registered). Pairs 1-3.
For each new condition X in (signrand, pzero) whose evaluation files exist, compared with the same pair's 80% (v13_s1_mixed)
and no-experience (v12_s1_random) runs (saved values; same code and GPU type, Sec.15):
  V1 : G5b, sign-following part (freeze - flip)/2 on green / pref: X - 80% < 0 per l1 late-5 point (eval_g5_v12.summ,
       per-episode pairs), 3 of 5 points -> pair, 2 of 3 pairs -> holds;
  V2 : true-pleasure Delta (A_cross protocol, l1 ep200, original green / pref): X - 80% < 0 (episode bootstrap,
       default_rng(0)), 2 of 3 pairs, per set (both -> holds, one -> partly);
  V3 : d_P = (g_hold - g_flip)/2 (G3, K = 15, mse late-5): X - 80% < 0, as V1;
  P1 : true-pleasure Delta: X - none > 0 (2 of 3 pairs) and V2 not holding; G3 hold: X - none > 0 per point (as V1).
  desc: X - none for all of the above, G1 tracking, G2 M1, G3 hold/zero/flip, BA (self-vision L1 l1 late-5, other-feature
       FP mse late-5, T = 50), self next-view error (visible / hidden).
-> data/result/paper_extra/C_signrand/summary.json
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11'), os.path.join(_PI, 'analyze', 'v14'), _PI):
    sys.path.insert(0, p)
from eval_g5_v12 import summ  # noqa
from aggregate_inverse_v14 import ba, cells, correct  # noqa

RES = os.path.join(_PI, 'data', 'result')
E = os.path.join(RES, 'paper_extra', 'C_signrand', 'eval')
EPS = (160, 170, 180, 190, 200)


def g5(c, s, e, q):
    if c in ('mixed80', 'random'):
        return np.load(os.path.join(RES, 'v13', 'main', 'g5', f's{s}', f'g5_per_ep{e}.npz'))[f'{"mixed" if c == "mixed80" else "random"}__{q}']
    return np.load(os.path.join(E, 'g5', c, f's{s}', f'g5_per_ep{e}.npz'))[f'{c}__{q}']


def flip(c, s, e, k):
    d = os.path.join(RES, 'v13', 'post', 'g5a_decomp', f's{s}') if c in ('mixed80', 'random') else \
        os.path.join(E, 'decomp', 'v13', 'post', 'g5a_decomp', f's{s}')
    return np.load(os.path.join(d, f'decomp_{c}_ep{e}.npz'))[f'{k}_flip_g']


def sf(c, s, e, k):
    return (g5(c, s, e, f'G5a_{k}_K15_freeze') - flip(c, s, e, k)) / 2


def delta(c, s, k):
    f = os.path.join(RES, 'paper_extra', 'A_cross', 'eval', f'{c}_s{s}_ep200_{k}.npz') if c in ('mixed80', 'random') else \
        os.path.join(E, 'd6', f's{s}', f'{c}_s{s}_ep200_{k}.npz')
    z = np.load(f)
    return z['true_fp'][:, 1:].mean(1) - z['zero_fp'][:, 1:].mean(1)


def g3(c, s, e, q):
    return np.load(os.path.join(E, 'g3flip', f'{c}_s{s}_ep{e}.npz'))[q]


def boot(x, n=1000):
    rng = np.random.default_rng(0)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return {'mean': float(x.mean()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def neg(r):
    return r['mean'] < 0 and r['ci95'][1] < 0


def pos(r):
    return r['mean'] > 0 and r['ci95'][0] > 0


def pointwise(fn, X, ref, sign):
    pts = [summ(fn(X, e) - fn(ref, e)) for e in EPS]
    ok = [neg(p) if sign < 0 else pos(p) for p in pts]
    return {'points': pts, 'n_ok': int(sum(ok)), 'pair_ok': sum(ok) >= 3}


def trial_json(c, s, e):
    if c == 'mixed80':
        f = os.path.join(RES, 'v13', 'main', 's1eval', f's{s}', f'eval_mixed_ep{e}.json')
    elif c == 'random':
        f = os.path.join(RES, 'v13', 'main', 's1eval', f's{s}', f'eval_random_ep{e}.json')
    else:
        f = os.path.join(E, 's1eval', c, f's{s}', f'eval_{c}_ep{e}.json')
    return json.load(open(f))


def inv(c, s, e, crit):
    if c == 'mixed80':
        f = os.path.join(RES, 'v14', 'inverse', f'{"v13" if crit == "L1" else "v13mse"}_mixed_{s}_ep{e}_pref.npz')
    else:
        fam = {'signrand': 'csr', 'pzero': 'cpz'}[c] + ('' if crit == 'L1' else 'mse')
        f = os.path.join(E, 'inverse', f'{fam}_{c}_{s}_ep{e}_pref.npz')
    z = np.load(f)
    return ba(correct(z, crit, 50), cells(z))


def available(c, s):
    return os.path.exists(os.path.join(E, 'logs', f'{c}_s{s}.done'))


def main():
    out = {}
    for X in ('signrand', 'pzero'):
        pairs = [s for s in range(3) if available(X, s)]
        if not pairs:
            continue
        R = {'pairs_available': pairs, 'per_pair': {}}
        for s in pairs:
            P = {}
            for ref, nm in (('mixed80', 'vs80'), ('random', 'vsnone')):
                sign = -1 if ref == 'mixed80' else 1
                P[f'G5b_{nm}'] = pointwise(lambda c, e: g5(c, s, e, 'G5b'), X, ref, sign)
                for k in ('green', 'pref'):
                    P[f'signfollow_{k}_{nm}'] = pointwise(lambda c, e, k=k: sf(c, s, e, k), X, ref, sign)
                P[f'dP_{nm}'] = pointwise(lambda c, e: g3(c, s, e, 'dP'), X, ref, sign)
                P[f'G3hold_{nm}'] = pointwise(lambda c, e: g3(c, s, e, 'hold'), X, ref, sign)
                for k in ('green', 'pref'):
                    b = boot(delta(X, s, k) - delta(ref, s, k))
                    P[f'delta_{k}_{nm}'] = {'diff': b, 'ok': neg(b) if sign < 0 else pos(b)}
            lv = {}
            for c in (X, 'mixed80', 'random'):
                d = {'G5b': [float(np.nanmean(g5(c, s, e, 'G5b'))) for e in EPS],
                     'signfollow_green': [float(np.nanmean(sf(c, s, e, 'green'))) for e in EPS],
                     'signfollow_pref': [float(np.nanmean(sf(c, s, e, 'pref'))) for e in EPS],
                     'delta_green': float(delta(c, s, 'green').mean()), 'delta_pref': float(delta(c, s, 'pref').mean())}
                for q in ('hold', 'zero', 'flip', 'dP'):
                    d[f'G3_{q}'] = [float(np.nanmean(g3(c, s, e, q))) for e in EPS]
                js = [trial_json(c, s, e) for e in EPS]
                d['G1_visible'] = [j['tracking']['p_next']['visible'] if 'visible' in j['tracking']['p_next'] else None for j in js]
                d['self_err_visible'] = [j['err_mean_visible'] for j in js]
                d['self_err_hidden'] = [j['err_mean_hidden'] for j in js]
                d['G2_M1'] = js[-1]['M1']
                if c != 'random':
                    d['BA_L1_l1late5'] = [inv(c, s, e, 'L1') for e in EPS]
                    d['BA_FP_mselate5'] = [inv(c, s, e, 'FP') for e in EPS]
                lv[c] = d
            P['values'] = lv
            reps = {}
            a = np.load(os.path.join(E, 'g5', X, f's{s}', 'g5_per_ep200.npz'))
            b = np.load(os.path.join(E, 'g5_rep', X, f's{s}', 'g5_per_ep200.npz'))
            reps['g5'] = all(np.array_equal(a[q], b[q], equal_nan=True) for q in a.files)
            ja, jb = trial_json(X, s, 200), json.load(open(os.path.join(E, 's1eval', X, f's{s}', f'eval_{X}_ep200_rep.json')))
            reps['trial_json'] = ja == jb
            P['rep_bit_identical'] = reps
            R['per_pair'][f's{s}'] = P
        J = {}
        for q in ('G5b', 'signfollow_green', 'signfollow_pref', 'dP', 'G3hold'):
            for nm in ('vs80', 'vsnone'):
                J[f'{q}_{nm}'] = sum(R['per_pair'][f's{s}'][f'{q}_{nm}']['pair_ok'] for s in pairs)
        for k in ('green', 'pref'):
            for nm in ('vs80', 'vsnone'):
                J[f'delta_{k}_{nm}'] = sum(R['per_pair'][f's{s}'][f'delta_{k}_{nm}']['ok'] for s in pairs)
        need = 2 if len(pairs) == 3 else len(pairs)
        v2 = [J[f'delta_{k}_vs80'] >= need for k in ('green', 'pref')]
        R['judge_counts'] = J
        R['verdict'] = {'V1_G5b': J['G5b_vs80'] >= need, 'V1_signfollow_green': J['signfollow_green_vs80'] >= need,
                        'V1_signfollow_pref': J['signfollow_pref_vs80'] >= need,
                        'V2': 'holds' if all(v2) else ('partly' if any(v2) else 'no'),
                        'V3': J['dP_vs80'] >= need,
                        'P1_delta': {k: J[f'delta_{k}_vsnone'] >= need and not v for k, v in zip(('green', 'pref'), v2)},
                        'P1_G3hold': J['G3hold_vsnone'] >= need, 'n_pairs': len(pairs)}
        out[X] = R
    with open(os.path.join(RES, 'paper_extra', 'C_signrand', 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    for X, R in out.items():
        print('==', X, 'pairs', R['pairs_available'], json.dumps(R['verdict']), json.dumps(R['judge_counts']))
        for s, P in R['per_pair'].items():
            print(' ', s, 'rep', P['rep_bit_identical'])
            for c, d in P['values'].items():
                print('   ', f'{c:9s}', {q: (round(float(np.mean(v)), 3) if isinstance(v, list) and v and v[0] is not None else (round(v, 3) if isinstance(v, float) else '')) for q, v in d.items() if q != 'G2_M1'})


if __name__ == '__main__':
    main()
