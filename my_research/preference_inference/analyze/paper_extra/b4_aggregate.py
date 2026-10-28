"""
Paper extra B4, judgement and description (docs/paper_extra_log.md Sec.29; pre-registered). Pairs 1-3.
On the test data made like A-1's own 80% stage-1 data (b4_run.py):
  B4-V1 : G5b (v13_s1_mixed80 test) and the sign-following part (freeze - flip)/2 on the new green / pref sets:
          signrand - 80% < 0 per l1 late-5 point (eval_g5_v12.summ, per-episode pairs; CI excludes 0),
          3 of 5 points -> pair, 2 of 3 pairs -> holds (the form of V1, Sec.4.4);
  desc  : signrand - none and 80% - none (same form), late-5 means of every model, frac_read_ctrl (G5b),
          frac_read of G5a, tracking R^2 (h1 -> next A-1 position, visible / hidden steps, mse late-5) on the
          v13_s1_mixed80 test and the common-test check against the saved trial evaluation; the same contrasts on
          the common test from C_signrand/summary.json for side-by-side reading.
  rep   : G5b of b4_run g5b and of b4_run g5 (independent runs) bit-identical at every point.
-> data/result/paper_extra/B4/summary.json
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)
from eval_g5_v12 import summ  # noqa

RES = os.path.join(_PI, 'data', 'result')
B4 = os.path.join(RES, 'paper_extra', 'B4')
EPS = (160, 170, 180, 190, 200)
CONDS = ('signrand', 'mixed80', 'random')


def g5b(c, s, e):
    return np.load(os.path.join(B4, 'g5b', f'{c}_s{s}_ep{e}.npz'))['G5b']


def g5full(c, s, e, q):
    return np.load(os.path.join(B4, 'g5', c, f's{s}', f'g5_per_ep{e}.npz'))[f'{c}__{q}']


def flip(c, s, e, k):
    return np.load(os.path.join(B4, 'decomp', 'v13', 'post', 'g5a_decomp', f's{s}', f'decomp_{c}_ep{e}.npz'))[f'{k}_flip_g']


def sf(c, s, e, k):
    return (g5full(c, s, e, f'G5a_{k}_K15_freeze') - flip(c, s, e, k)) / 2


def neg(r):
    return r['mean'] < 0 and r['ci95'][1] < 0


def pos(r):
    return r['mean'] > 0 and r['ci95'][0] > 0


def pointwise(fn, X, ref, sign):
    pts = [summ(fn(X, e) - fn(ref, e)) for e in EPS]
    ok = [neg(p) if sign < 0 else pos(p) for p in pts]
    return {'points': pts, 'n_ok': int(sum(ok)), 'pair_ok': sum(ok) >= 3}


def main():
    QS = {'G5b': lambda s: (lambda c, e: g5b(c, s, e)),
          'signfollow_green': lambda s: (lambda c, e: sf(c, s, e, 'green')),
          'signfollow_pref': lambda s: (lambda c, e: sf(c, s, e, 'pref'))}
    common = json.load(open(os.path.join(RES, 'paper_extra', 'C_signrand', 'summary.json')))['signrand']['per_pair']
    R = {'per_pair': {}}
    for s in range(3):
        P = {}
        for q, mk in QS.items():
            fn = mk(s)
            P[f'{q}_sr_vs80'] = pointwise(fn, 'signrand', 'mixed80', -1)
            P[f'{q}_sr_vsnone'] = pointwise(fn, 'signrand', 'random', 1)
            P[f'{q}_80_vsnone'] = pointwise(fn, 'mixed80', 'random', 1)
        lv = {}
        for c in CONDS:
            d = {q: [float(np.nanmean(QS[q](s)(c, e))) for e in EPS] for q in QS}
            d['frac_read_ctrl'] = [json.load(open(os.path.join(B4, 'g5b', f'{c}_s{s}_ep{e}.json')))['frac_read_ctrl'] for e in EPS]
            j = [json.load(open(os.path.join(B4, 'g5', c, f's{s}', f'g5_{c}_ep{e}.json'))) for e in EPS]
            d['G5a_frac_read'] = {k: [x['G5a'][f'{k}_K15_freeze']['frac_read'] for x in j] for k in ('green', 'pref')}
            tr = [json.load(open(os.path.join(B4, 'track', f'{c}_s{s}_ep{e}.json'))) for e in EPS]
            d['track_visible'] = [t['p_next']['visible'] for t in tr]
            d['track_hidden'] = [t['p_next']['hidden'] for t in tr]
            d['common_test'] = {q: common[f's{s}']['values'][c][q] for q in ('G5b', 'signfollow_green', 'signfollow_pref', 'G1_visible')}
            lv[c] = d
        P['values'] = lv
        P['rep_G5b_bit_identical'] = all(np.array_equal(g5b(c, s, e), g5full(c, s, e, 'G5b'), equal_nan=True)
                                         for c in CONDS for e in EPS)
        chk = {}
        for c in CONDS:
            f = os.path.join(B4, 'track_common', f'{c}_s{s}_ep200.json')
            if os.path.exists(f):
                chk[c] = {'b4_common': json.load(open(f))['p_next']['visible'],
                          'saved': common[f's{s}']['values'][c]['G1_visible'][-1]}
        P['track_common_check'] = chk
        R['per_pair'][f's{s}'] = P
    J = {k: sum(R['per_pair'][f's{s}'][k]['pair_ok'] for s in range(3)) for k in R['per_pair']['s0'] if 'points' in R['per_pair']['s0'][k]}
    R['judge_counts'] = J
    R['verdict'] = {f'B4V1_{q}': J[f'{q}_sr_vs80'] >= 2 for q in QS}
    with open(os.path.join(B4, 'summary.json'), 'w') as f:
        json.dump(R, f, indent=1)
    print(json.dumps(R['verdict']), json.dumps(J))
    for s, P in R['per_pair'].items():
        print(s, 'rep', P['rep_G5b_bit_identical'], 'track check', P['track_common_check'])
        for c, d in P['values'].items():
            print('  ', f'{c:9s}', {q: round(float(np.mean(v)), 3) for q, v in d.items() if isinstance(v, list)})


if __name__ == '__main__':
    main()
