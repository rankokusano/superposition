"""
Paper extra B, checks outside the pre-registration (docs/paper_extra_log.md Sec.20; method recorded before computing). CPU.
  20.1 BA (T = 50) of copies into the no-experience v13-NS model: full copy (v13nscp:random), pleasure columns
       (v13nscpp:random), and the full copy of the experienced model (v13nscp:mixed, re-measured on CPU); self-vision L1
       on the l1 late-5, other-feature FP on the mse late-5; the original no-experience NS from the saved GPU files.
       Difference full-copy experienced - full-copy no-experience: aggregate_inverse_v14.boot indices (RandomState(0),
       1000, all four cells), per point and for the late-5 mean.
  20.2 true-pleasure Delta (true - zero, other-side feature-prediction MSE, t = 1..99, l1 ep200) for ns / nscp / nscpp
       (experienced), ns / nscp (no experience) and 80%, on green / pref at A-2 0.5 and 0.8; episode bootstrap CIs
       (default_rng(0)); D = Delta(0.8) - Delta(0.5); copy - original; experienced - no experience.
       Check: 80% on CPU vs A's GPU values.
-> data/result/paper_extra/B_extra/summary.json
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v14'))
from aggregate_inverse_v14 import ba, boot as cellboot, cells, correct, ci  # noqa

RES = os.path.join(_PI, 'data', 'result')
X = os.path.join(RES, 'paper_extra', 'B_extra')
EPS = (160, 170, 180, 190, 200)
SETS = ('green', 'pref', 'green_a2p80', 'pref_a2p80')
D6 = ('ns_mixed', 'nscp_mixed', 'nscpp_mixed', 'ns_random', 'nscp_random', 'mixed80')


def inv(fam, c, s, e, saved=False):
    d = os.path.join(RES, 'v14', 'inverse') if saved else os.path.join(X, 'inverse')
    return np.load(os.path.join(d, f'{fam}_{c}_{s}_ep{e}_pref.npz'))


def ba_block(s):
    out = {}
    specs = {'nscp_mixed': ('v13nscp', 'v13nscpmse', 'mixed', False), 'nscp_random': ('v13nscp', 'v13nscpmse', 'random', False),
             'nscpp_random': ('v13nscpp', 'v13nscppmse', 'random', False), 'ns_random_gpu': ('v13ns', 'v13nsmse', 'random', True)}
    C = {}
    for nm, (fl, fm, c, sv) in specs.items():
        for crit, fam in (('L1', fl), ('FP', fm)):
            zs = [inv(fam, c, s, e, sv) for e in EPS]
            C[(nm, crit)] = ([correct(z, crit, 50) for z in zs], cells(zs[0]))
            out.setdefault(nm, {})[f'BA_{crit}'] = [ba(cc, C[(nm, crit)][1]) for cc in C[(nm, crit)][0]]
    idx = cellboot(cells(inv('v13nscp', 'mixed', s, 200)), 300)
    for crit in ('L1', 'FP'):
        (ca, cl), (cb, _) = C[('nscp_mixed', crit)], C[('nscp_random', crit)]
        pts = []
        for a, b in zip(ca, cb):
            bs = [ba(a, cl, i) - ba(b, cl, i) for i in idx]
            pts.append({'diff': ba(a, cl) - ba(b, cl), 'ci95': ci(bs)})
        mean_bs = [np.mean([ba(a, cl, i) - ba(b, cl, i) for a, b in zip(ca, cb)]) for i in idx]
        out[f'diff_nscp_mixed_minus_nscp_random_{crit}'] = {'points': pts, 'late5_mean': float(np.mean([p['diff'] for p in pts])),
                                                            'late5_ci95': ci(mean_bs)}
    gpu = json.load(open(os.path.join(RES, 'paper_extra', 'B_copy', 'summary.json')))['per_pair'][f's{s}']['desc']['copy']
    out['nscp_mixed_gpu'] = {'BA_L1': gpu['BA_L1_l1late5'], 'BA_FP': gpu['BA_FP_mselate5']}
    return out


def dfp(m, s, k, gpu_a=False):
    f = os.path.join(RES, 'paper_extra', 'A_cross', 'eval', f'{m}_s{s}_ep200_{k}.npz') if gpu_a else \
        os.path.join(X, 'd6', 'cpu', f'{m}_s{s}_ep200_{k}.npz')
    z = np.load(f)
    return z['true_fp'][:, 1:].mean(1) - z['zero_fp'][:, 1:].mean(1), z


def bt(x, n=1000):
    rng = np.random.default_rng(0)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return {'mean': float(x.mean()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def d6_block(s):
    out = {'Delta': {}, 'D_08_minus_05': {}, 'contrasts': {}}
    V = {(m, k): dfp(m, s, k)[0] for m in D6 for k in SETS}
    for (m, k), v in V.items():
        out['Delta'][f'{m}_{k}'] = bt(v)
    for m in D6:
        for k in ('green', 'pref'):
            out['D_08_minus_05'][f'{m}_{k}'] = bt(V[(m, k + '_a2p80')] - V[(m, k)])
    for k in SETS:
        for a, b in (('nscp_mixed', 'ns_mixed'), ('nscpp_mixed', 'ns_mixed'), ('nscp_random', 'ns_random'),
                     ('nscp_mixed', 'nscp_random'), ('ns_mixed', 'ns_random'), ('nscp_mixed', 'mixed80')):
            out['contrasts'][f'{a}_minus_{b}_{k}'] = bt(V[(a, k)] - V[(b, k)])
    chk = {}
    for k in SETS:
        _, zc = dfp('mixed80', s, k)
        _, zg = dfp('mixed80', s, k, gpu_a=True)
        chk[k] = {q: float(np.nanmax(np.abs(zc[q] - zg[q]))) for q in ('zero_fp', 'true_fp')}
    out['check_mixed80_cpu_vs_A_gpu_maxabs'] = chk
    return out


def main():
    res = {f's{s}': {'BA': ba_block(s), 'd6': d6_block(s)} for s in range(3)}
    with open(os.path.join(X, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1)
    for s in range(3):
        r = res[f's{s}']
        print(f'== pair {s + 1}')
        for nm in ('ns_random_gpu', 'nscp_random', 'nscpp_random', 'nscp_mixed', 'nscp_mixed_gpu'):
            print(f'  {nm:14s} L1 {np.round(r["BA"][nm]["BA_L1"], 3)}  FP {np.round(r["BA"][nm]["BA_FP"], 3)}')
        for crit in ('L1', 'FP'):
            d = r['BA'][f'diff_nscp_mixed_minus_nscp_random_{crit}']
            print(f'  diff exp-noexp (full copy) {crit}: late5 {d["late5_mean"]:+.3f} {np.round(d["late5_ci95"], 3)}')
        for k in SETS:
            print(f'  Delta {k:12s}', ' '.join(f'{m} {r["d6"]["Delta"][f"{m}_{k}"]["mean"]:+.3f}' for m in D6))
        print('  D(0.8-0.5)', {q: round(v['mean'], 3) for q, v in r['d6']['D_08_minus_05'].items()})
        print('  check 80% cpu vs gpu', r['d6']['check_mixed80_cpu_vs_A_gpu_maxabs'])


if __name__ == '__main__':
    main()
