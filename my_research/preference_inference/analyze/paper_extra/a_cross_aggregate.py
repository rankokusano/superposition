"""
Paper extra A, checks and statistics (docs/paper_extra_log.md Sec.2; pre-registered before computing).
Reads data/result/paper_extra/A_cross/eval/<cond>_s<s>_ep200_<set>.npz (a_cross_eval.py).
  checks : eval_rep (second run of mixed80 s1) bit-identical to eval; p = 0.5 values vs the saved GPU values of the
           D6-like secondary (eval_d6_v12 per-episode A - B of fp / self / a2err, and the json means of A and B fp);
  stats  : per episode Delta_e = mean_{t=1..99}(fp_true - fp_zero); per pair s the contrasts (Sec.2.4)
             D_A1 = D80(0.8) - D80(0.5), R = D80(0.5) - D50(0.5), I = [D80 - D50](0.8) - [D80 - D50](0.5),
             J = D_A1 - D_rand, D_rand, D_50, and D80(0.8) - D50(0.8),
           with the paired episode bootstrap (np.random.default_rng(0), 1000, 95%); judgement on pairs 1-3
           (2 of 3 with CI excluding 0 in the predicted direction, per set); over 6 pairs the paired t 95% CI (df 5)
           and the count of pairs in the predicted direction with CI excluding 0;
  split  : (b) and Delta by A-2's step a2_source[t] (1 follow, 0/2 not follow, 3 sleep), t = 1..99, per episode
           mean over that kind of frame, then over episodes;
  also   : (a), (b), A-2 position error (read from the predicted image), self-vision error.
-> data/result/paper_extra/A_cross/summary.json
"""

import json
import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(_PI, 'data', 'result')
A = os.path.join(RES, 'paper_extra', 'A_cross')
CONDS = ('random', 'mixed50', 'mixed80')
SEEDS = range(6)
SETS = {'green': ('green', 'green_a2p80'), 'pref': ('pref', 'pref_a2p80')}
SAVED = {('random', s): os.path.join(RES, 'v13', 'main', 'd6', f's{s}', 'd6_random_ep200') for s in SEEDS}
SAVED.update({('mixed80', s): os.path.join(RES, 'v13', 'main', 'd6', f's{s}', 'd6_mixed_ep200') for s in SEEDS})
SAVED.update({('mixed50', s): os.path.join(RES, 'v12', 'main', 'd6', f's{s}', 'd6_mixed_ep200') for s in range(4)})
SAVED.update({('mixed50', s): os.path.join(RES, 'v13', 'v12ref', 'd6', f's{s}', 'd6_mixed_ep200') for s in (4, 5)})


def load(c, s, k, sub='eval'):
    return np.load(os.path.join(A, sub, f'{c}_s{s}_ep200_{k}.npz'))


def delta(z):
    return z['true_fp'][:, 1:].mean(1) - z['zero_fp'][:, 1:].mean(1)


def boot(x, n=1000):
    rng = np.random.default_rng(0)
    N = len(x)
    bs = np.array([x[rng.integers(0, N, N)].mean() for _ in range(n)])
    return {'mean': float(x.mean()), 'ci95': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def sig(r, sign):
    lo, hi = r['ci95']
    return bool(hi < 0) if sign < 0 else bool(lo > 0)


def checks():
    out = {}
    eq = {}
    for k in ('green', 'pref', 'green_a2p80', 'pref_a2p80'):
        a, b = load('mixed80', 1, k), load('mixed80', 1, k, 'eval_rep')
        eq[k] = all(np.array_equal(a[q], b[q], equal_nan=True) for q in a.files)
    out['rep_bit_identical_mixed80_s1'] = eq
    diffs = {}
    for (c, s), path in SAVED.items():
        per = np.load(path + '_per.npz')
        js = json.load(open(path + '.json'))
        for k in ('green', 'pref'):
            z = load(c, s, k)
            dfp = z['zero_fp'][:, 1:].mean(1) - z['true_fp'][:, 1:].mean(1)
            dsl = z['zero_self'][:, 1:].mean(1) - z['true_self'][:, 1:].mean(1)
            da2 = np.nanmean(z['zero_a2err'][:, 1:], 1) - np.nanmean(z['true_a2err'][:, 1:], 1)
            diffs[f'{c}_s{s}_{k}'] = {
                'perep_A_minus_B_fp': float(np.abs(dfp - per[f'{k}_fp']).max()),
                'perep_A_minus_B_self': float(np.abs(dsl - per[f'{k}_self']).max()),
                'perep_A_minus_B_a2err': float(np.nanmax(np.abs(da2 - per[f'{k}_a2err']))),
                'mean_fp_A': abs(float(z['zero_fp'][:, 1:].mean()) - js[k]['A']['fp']),
                'mean_fp_B': abs(float(z['true_fp'][:, 1:].mean()) - js[k]['B']['fp'])}
    out['vs_saved_gpu_max'] = {q: max(v[q] for v in diffs.values()) for q in next(iter(diffs.values()))}
    out['vs_saved_gpu'] = diffs
    return out


def contrasts(s, k):
    k5, k8 = SETS[k]
    D = {(c, p): delta(load(c, s, kk)) for c in CONDS for p, kk in ((0.5, k5), (0.8, k8))}
    a1 = D[('mixed80', 0.8)] - D[('mixed80', 0.5)]
    dr = D[('random', 0.8)] - D[('random', 0.5)]
    d50 = D[('mixed50', 0.8)] - D[('mixed50', 0.5)]
    c = {'D_A1': a1, 'R': D[('mixed80', 0.5)] - D[('mixed50', 0.5)],
         'I': (D[('mixed80', 0.8)] - D[('mixed50', 0.8)]) - (D[('mixed80', 0.5)] - D[('mixed50', 0.5)]),
         'J': a1 - dr, 'D_rand': dr, 'D_50': d50, 'D80_minus_D50_at08': D[('mixed80', 0.8)] - D[('mixed50', 0.8)]}
    return {q: boot(v) for q, v in c.items()}, {f'{cc}_{p}': boot(v) for (cc, p), v in D.items()}


def split(z):
    src = z['a2_source'][:, 1:100]
    out = {}
    for nm, kinds in (('follow', [1]), ('notfollow', [0, 2]), ('sleep', [3])):
        m = np.isin(src, kinds)
        def epm(x):  # noqa
            x = x[:, 1:100]
            w = m.sum(1)
            v = np.where(w > 0, (x * m).sum(1) / np.maximum(w, 1), np.nan)
            return v
        b = epm(z['true_fp']); d = b - epm(z['zero_fp'])
        out[nm] = {'share': float(m.mean()), 'true_fp': float(np.nanmean(b)), 'delta': float(np.nanmean(d))}
    return out


def main():
    res = {'checks': checks()}
    PRED = {'D_A1': -1, 'I': -1, 'J': -1, 'R': 1}
    per = {}
    for k in SETS:
        for s in SEEDS:
            cts, D = contrasts(s, k)
            per[(k, s)] = cts
            res.setdefault('per_pair', {})[f'{k}_s{s}'] = {'contrasts': cts, 'Delta': D}
    judge = {}
    for k in SETS:
        for q, sg in PRED.items():
            ok3 = [sig(per[(k, s)][q], sg) for s in range(3)]
            ok6 = [sig(per[(k, s)][q], sg) for s in SEEDS]
            m6 = np.array([per[(k, s)][q]['mean'] for s in SEEDS])
            t = stats.t.ppf(0.975, 5) * m6.std(ddof=1) / np.sqrt(6)
            judge[f'{k}_{q}'] = {'pairs1_3_ok': ok3, 'supported_main': sum(ok3) >= 2, 'pairs1_6_ok_count': int(sum(ok6)),
                                 'mean6': float(m6.mean()), 't_ci95_6': [float(m6.mean() - t), float(m6.mean() + t)],
                                 'per_pair_means': m6.round(5).tolist()}
    res['judge'] = judge
    for q in ('D_A1', 'I', 'J'):
        g, p = judge[f'green_{q}']['supported_main'], judge[f'pref_{q}']['supported_main']
        res.setdefault('verdict', {})[q] = 'supported' if g and p else ('partly supported' if g or p else 'not supported')
    res['split'] = {f'{c}_s{s}_{kk}': split(load(c, s, kk)) for c in CONDS for s in SEEDS
                    for kk in ('green', 'pref', 'green_a2p80', 'pref_a2p80')}
    res['secondary'] = {f'{c}_s{s}_{kk}': {v: {'fp': float(z[f'{v}_fp'][:, 1:].mean()), 'self': float(z[f'{v}_self'][:, 1:].mean()),
                                               'a2err': float(np.nanmean(z[f'{v}_a2err'][:, 1:])),
                                               'a2_frac_read': float(np.isfinite(z[f'{v}_a2err'][:, 1:]).mean())}
                                           for v in ('zero', 'true')}
                        for c in CONDS for s in SEEDS for kk in ('green', 'pref', 'green_a2p80', 'pref_a2p80')
                        for z in [load(c, s, kk)]}
    with open(os.path.join(A, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({'checks': {k: v for k, v in res['checks'].items() if k != 'vs_saved_gpu'}, 'verdict': res['verdict'],
                      'judge': res['judge']}, indent=1))


if __name__ == '__main__':
    main()
