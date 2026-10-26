"""
Paper extra B, judgement and description (docs/paper_extra_log.md Sec.3, 13; pre-registered). Pairs 1-3.
Models: NS = v13ns_s1_mixed (saved values, re-measure bit-identical, Sec.15), copy = all of lstm_other <- lstm,
pcols = pleasure input columns only (Sec.13), 80% = v13_s1_mixed, none = v12_s1_random (saved), and as a check outside
the pre-registration the copy of the NS no-experience model (copy-NSrand).
  B1 / B3 : per l1 late-5 point, G5b of copy (pcols) > 0, - none > 0, - NS > 0 (each CI excludes 0; eval_g5_v12.summ,
            per-episode pairs on the common test); 3 of 5 points -> pair; 2 of 3 pairs -> supported. A pair whose
            h2 -> A-2 position R^2 (x or y) < 0.8 is "not readable" and not counted.
  B2      : G5b, sign-following parts (freeze - flip)/2 green/pref, BA (self-vision L1, l1 late-5; other-feature FP,
            mse late-5; T = 50): late-5 means per model, and the count of points where |model - 80%| < |NS - 80%|.
  qty 4   : h2 R^2 (x, y), frac_read_ctrl (G5b control), A-2 position error with zero input (green set).
  qty 5   : G3 (freeze K15 hold, mse late-5) and the self next-view error (visible / hidden), mse late-5.
  check   : pcols h2 and G1-G3 outputs bit-identical to NS (Sec.13.2).
-> data/result/paper_extra/B_copy/summary.json
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v14'))
sys.path.insert(0, _PI)
from eval_g5_v12 import summ  # noqa
from aggregate_inverse_v14 import ba, cells, correct  # noqa

RES = os.path.join(_PI, 'data', 'result')
PX = os.path.join(RES, 'paper_extra')
EPS = (160, 170, 180, 190, 200)
G5 = {'NS': ('v13/ns/g5', 'mixed'), 'copy': ('paper_extra/B_copy/g5', 'mixed'), 'pcols': ('paper_extra/B_copy_pcols/g5', 'mixed'),
      '80': ('v13/main/g5', 'mixed'), 'none': ('v13/main/g5', 'random'), 'copy_NSrand': ('paper_extra/B_copy/g5', 'random'),
      'NSrand': ('v13/ns/g5', 'random'), 'pcols_NSrand': ('paper_extra/B_copy_pcols/g5', 'random')}
DEC = {'NS': ('v13/post/g5a_decomp_ns', 'mixed80'), 'copy': ('paper_extra/B_copy/decomp/v13/post/g5a_decomp', 'mixed80'),
       'pcols': ('paper_extra/B_copy_pcols/decomp/v13/post/g5a_decomp', 'mixed80'), '80': ('v13/post/g5a_decomp', 'mixed80')}
INV = {'NS': ('v14/inverse', 'v13ns', 'v13nsmse'), 'copy': ('paper_extra/B_copy/inverse', 'v13nscp', 'v13nscpmse'),
       'pcols': ('paper_extra/B_copy_pcols/inverse', 'v13nscpp', 'v13nscppmse'), '80': ('v14/inverse', 'v13', 'v13mse')}
H2 = {'NS': 'v13/ns/h2_check_s{s}', 'copy': 'paper_extra/B_copy/h2_check_s{s}', 'pcols': 'paper_extra/B_copy_pcols/h2_check_s{s}',
      '80': 'v13/main/h2_check_s{s}'}
S1 = {'NS': 'v13/ns/s1eval/s{s}', 'copy': 'paper_extra/B_copy/s1eval/s{s}', 'pcols': 'paper_extra/B_copy_pcols/s1eval/s{s}'}


def g5(m, s, e, q='G5b'):
    sub, c = G5[m]
    return np.load(os.path.join(RES, sub, f's{s}', f'g5_per_ep{e}.npz'))[f'{c}__{q}']


def g5json(m, s, e):
    sub, c = G5[m]
    return json.load(open(os.path.join(RES, sub, f's{s}', f'g5_{c}_ep{e}.json')))


def signfollow(m, s, e, k):
    sub, c = DEC[m]
    fl = np.load(os.path.join(RES, sub, f's{s}', f'decomp_{c}_ep{e}.npz'))[f'{k}_flip_g']
    return (g5(m, s, e, f'G5a_{k}_K15_freeze') - fl) / 2


def pos(r):
    return r['mean'] > 0 and r['ci95'][0] > 0


def main():
    out = {'per_pair': {}}
    for s in range(3):
        P = {}
        h2 = {m: json.load(open(os.path.join(RES, H2[m].format(s=s), 'h2_check.json')))[f'mixed:{s}:200']['ridge_xy'] for m in H2}
        h2['copy_NSrand'] = json.load(open(os.path.join(RES, H2['copy'].format(s=s), 'h2_check.json')))[f'random:{s}:200']['ridge_xy']
        P['h2_A2pos_r2_xy'] = h2
        readable = {m: min(v) >= 0.8 for m, v in h2.items()}
        P['readable'] = readable
        for test, m in (('B1', 'copy'), ('B3', 'pcols')):
            pts = []
            for e in EPS:
                a = g5(m, s, e)
                r = {'G5b': summ(a), 'minus_none': summ(a - g5('none', s, e)), 'minus_NS': summ(a - g5('NS', s, e))}
                r['ok'] = all(pos(v) for v in r.values())
                pts.append(r)
            P[test] = {'points': dict(zip(map(str, EPS), pts)), 'n_ok': sum(p['ok'] for p in pts),
                       'pair_ok': sum(p['ok'] for p in pts) >= 3, 'readable': readable[m]}
        desc = {}
        for m in ('NS', 'copy', 'pcols', '80', 'none', 'copy_NSrand', 'NSrand', 'pcols_NSrand'):
            v = [summ(g5(m, s, e)) for e in EPS]
            fr = [g5json(m, s, e)['G5b']['frac_read_ctrl'] for e in EPS]
            desc[m] = {'G5b_late5': [x['mean'] for x in v], 'G5b_ci_late5': [x['ci95'] for x in v], 'frac_read_ctrl': fr}
        for m in DEC:
            for k in ('green', 'pref'):
                desc[m][f'signfollow_{k}'] = [float(np.nanmean(signfollow(m, s, e, k))) for e in EPS]
        for m, (sub, fam, famm) in INV.items():
            L = []
            for e in EPS:
                z = np.load(os.path.join(RES, sub, f'{fam}_mixed_{s}_ep{e}_pref.npz'))
                L.append(ba(correct(z, 'L1', 50), cells(z)))
            F = []
            for e in EPS:
                z = np.load(os.path.join(RES, sub, f'{famm}_mixed_{s}_ep{e}_pref.npz'))
                F.append(ba(correct(z, 'FP', 50), cells(z)))
            desc[m]['BA_L1_l1late5'] = L
            desc[m]['BA_FP_mselate5'] = F
        for m, nm in (('copy', 'nscp'), ('pcols', 'nscpp'), ('NS', 'ns')):
            sub = 'B_copy_pcols' if m == 'pcols' else 'B_copy'
            f = os.path.join(PX, sub, 'd6', f's{s}', f'{nm}_mixed_s{s}_ep200_green.npz')
            if os.path.exists(f):
                z = np.load(f)
                desc[m]['a2err_zero_green'] = float(np.nanmean(z['zero_a2err'][:, 1:]))
                desc[m]['a2_frac_read_zero_green'] = float(np.isfinite(z['zero_a2err'][:, 1:]).mean())
        for m in S1:
            js = [json.load(open(os.path.join(RES, S1[m].format(s=s), f'eval_mixed_ep{e}.json'))) for e in EPS]
            desc[m]['G3_K15_hold_g'] = [j['freeze']['K15_hold']['g']['mean'] for j in js]
            desc[m]['self_err_visible'] = [j['err_mean_visible'] for j in js]
            desc[m]['self_err_hidden'] = [j['err_mean_hidden'] for j in js]
        js80 = [json.load(open(os.path.join(RES, 'v13', 'main', 's1eval', f's{s}', f'eval_mixed_ep{e}.json'))) for e in EPS] \
            if os.path.exists(os.path.join(RES, 'v13', 'main', 's1eval', f's{s}', 'eval_mixed_ep200.json')) else []
        if js80:
            desc['80']['G3_K15_hold_g'] = [j['freeze']['K15_hold']['g']['mean'] for j in js80]
            desc['80']['self_err_visible'] = [j['err_mean_visible'] for j in js80]
            desc['80']['self_err_hidden'] = [j['err_mean_hidden'] for j in js80]
        # B2: closer to 80% than NS is, per point
        b2 = {}
        for q in ('G5b_late5', 'signfollow_green', 'signfollow_pref', 'BA_L1_l1late5', 'BA_FP_mselate5'):
            for m in ('copy', 'pcols'):
                b2[f'{m}_{q}'] = int(sum(abs(a - c) < abs(n - c) for a, n, c in zip(desc[m][q], desc['NS'][q], desc['80'][q])))
        P['B2_closer_than_NS_points'] = b2
        P['desc'] = desc
        # check: pcols h2 / G1-G3 bit-identical to NS
        chk = {'h2_json_equal': json.load(open(os.path.join(RES, H2['pcols'].format(s=s), 'h2_check.json'))) ==
               json.load(open(os.path.join(RES, H2['NS'].format(s=s), 'h2_check.json')))}
        for e in EPS:
            a = np.load(os.path.join(RES, S1['pcols'].format(s=s), f'per_ep{e}.npz'))
            b = np.load(os.path.join(RES, S1['NS'].format(s=s), f'per_ep{e}.npz'))
            chk[f's1eval_per_ep{e}_equal'] = set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
            ja = json.load(open(os.path.join(RES, S1['pcols'].format(s=s), f'eval_mixed_ep{e}.json')))
            jb = json.load(open(os.path.join(RES, S1['NS'].format(s=s), f'eval_mixed_ep{e}.json')))
            ja.pop('exp'); jb.pop('exp')
            chk[f's1eval_json_ep{e}_equal'] = ja == jb
        rep = {}
        for sub in ('B_copy', 'B_copy_pcols'):
            a = np.load(os.path.join(PX, sub, 'g5', f's{s}', 'g5_per_ep200.npz'))
            b = np.load(os.path.join(PX, sub, 'g5_rep', f's{s}', 'g5_per_ep200.npz'))
            rep[sub] = all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
        chk['g5_rep_bit_identical'] = rep
        P['checks'] = chk
        out['per_pair'][f's{s}'] = P
    out['B1_supported'] = sum(out['per_pair'][f's{s}']['B1']['pair_ok'] for s in range(3)) >= 2
    out['B3_supported'] = sum(out['per_pair'][f's{s}']['B3']['pair_ok'] for s in range(3)) >= 2
    with open(os.path.join(PX, 'B_copy', 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    for s in range(3):
        P = out['per_pair'][f's{s}']
        print(f'== pair {s + 1}: h2 {json.dumps({m: [round(x, 3) for x in v] for m, v in P["h2_A2pos_r2_xy"].items()})}')
        print('  B1', P['B1']['n_ok'], P['B1']['pair_ok'], 'readable', P['B1']['readable'], '| B3', P['B3']['n_ok'], P['B3']['pair_ok'])
        for m, d in P['desc'].items():
            print(f'  {m:13s}', {q: ([round(x, 3) for x in v] if isinstance(v, list) and not isinstance(v[0], list) else (round(v, 3) if isinstance(v, float) else '')) for q, v in d.items() if q != 'G5b_ci_late5'})
        print('  B2', P['B2_closer_than_NS_points'])
        print('  checks', P['checks'])
    print('B1 supported', out['B1_supported'], 'B3 supported', out['B3_supported'])


if __name__ == '__main__':
    main()
