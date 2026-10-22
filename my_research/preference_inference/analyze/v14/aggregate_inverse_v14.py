"""
v14-A judgement (pre-registered in docs/v14_experiment_log.md Sec.3) from eval_inverse_v14.py outputs.
Per model, checkpoint and set: choose cp if cumulative error (target frames 1..T) under cp < under cm (tie 0.5),
correct = correct_sign. Balanced accuracy (BA) = mean of the per-cell accuracies over the cells present
(layout x A-2 liked; 4 cells on the pref set, 2 on the green set). 95% CI: 1000 episode bootstrap
(RandomState(0)); the paired difference (experienced - no-experience) uses the same resampled episodes.
A point passes if BA_exp > 0.5 with CI excluding 0.5 AND diff > 0 with CI excluding 0 (main: pref, L1, T = 50);
a pair passes with >= 3 of the 5 late-5 points; the gate needs 2 of 3 pairs (3 of 4 with s3).
Also: the same for every T, for the FP error, the green set, the c0 comparison and the reference (true P_A2)
upper bound; 'reverse' (no-experience BA < 0.5 with CI excluding 0.5) is flagged.
Usage: python analyze/v14/aggregate_inverse_v14.py --pairs v12:mixed:0,v12:random:0 ... [--sub inverse]
"""

import argparse
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
BASE = os.path.join(_PI, 'data', 'result', 'v14')
TS = (5, 10, 20, 30, 50, 100)
EPOCHS = (160, 170, 180, 190, 200)
N_BOOT = 1000


def cells(z):
    return z['layout'].astype(int) * 2 + z['a2_liked'].astype(int)


def choice(z, q, T, a='cp', b='cm'):
    """1 if a chosen, 0 if b, 0.5 on a tie."""
    ea, eb = z[f'{q}_{a}'][:, :T].sum(1), z[f'{q}_{b}'][:, :T].sum(1)
    return np.where(ea < eb, 1.0, np.where(ea > eb, 0.0, 0.5))


def correct(z, q, T):
    ch = choice(z, q, T)
    return np.where(z['correct_sign'] > 0, ch, 1 - ch)


def correct_ref(z, q, T):
    """reference: ref_red vs ref_green, correct = A-2's true liked colour."""
    ch = choice(z, q, T, 'ref_red', 'ref_green')
    return np.where(z['a2_liked'] == 1, ch, 1 - ch)


def ba(c, cl, idx=None):
    if idx is not None:
        c, cl = c[idx], cl[idx]
    return float(np.mean([c[cl == k].mean() for k in np.unique(cl)]))


def boot(cl, n, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < N_BOOT:
        idx = rng.randint(0, n, n)
        if len(np.unique(cl[idx])) == len(np.unique(cl)):   # keep every cell (never fails at N = 300)
            out.append(idx)
    return out


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def load(sub, spec, e, s):
    return np.load(os.path.join(BASE, sub, spec.replace(':', '_') + f'_ep{e}_{s}.npz'))


def point(ze, zr, q, T, B):
    cl = cells(ze)
    assert (cl == cells(zr)).all()
    ce, cr = correct(ze, q, T), correct(zr, q, T)
    be = np.array([ba(ce, cl, i) for i in B]); br = np.array([ba(cr, cl, i) for i in B])
    r = {'ba_exp': ba(ce, cl), 'ba_exp_ci': ci(be), 'ba_noexp': ba(cr, cl), 'ba_noexp_ci': ci(br),
         'diff': ba(ce, cl) - ba(cr, cl), 'diff_ci': ci(be - br),
         'raw_acc_exp': float(ce.mean()), 'raw_acc_noexp': float(cr.mean()),
         'cell_acc_exp': {int(k): float(ce[cl == k].mean()) for k in np.unique(cl)},
         'cell_acc_noexp': {int(k): float(cr[cl == k].mean()) for k in np.unique(cl)}}
    r['pass'] = bool(r['ba_exp_ci'][0] > 0.5 and r['diff_ci'][0] > 0)
    r['noexp_reverse'] = bool(r['ba_noexp_ci'][1] < 0.5)
    r['exp_reverse'] = bool(r['ba_exp_ci'][1] < 0.5)
    cref = correct_ref(ze, q, T)
    r['ref_ba_exp'] = ba(cref, cl)
    r['ref_ba_noexp'] = ba(correct_ref(zr, q, T), cl)
    r['effect_ratio_exp'] = (r['ba_exp'] - 0.5) / (r['ref_ba_exp'] - 0.5) if r['ref_ba_exp'] != 0.5 else None
    # c0: is the correct candidate's cumulative error below the no-preference candidate's?
    for nm, z in (('exp', ze), ('noexp', zr)):
        ecor = np.where(z['correct_sign'] > 0, z[f'{q}_cp'][:, :T].sum(1), z[f'{q}_cm'][:, :T].sum(1))
        d0 = z[f'{q}_c0'][:, :T].sum(1) - ecor
        r[f'c0_minus_correct_{nm}'] = {'mean': float(d0.mean()), 'ci': ci([d0[i].mean() for i in B]),
                                       'frac_correct_better': float((d0 > 0).mean())}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', nargs='+', required=True, help='exp_spec,noexp_spec')
    ap.add_argument('--sub', default='inverse')
    ap.add_argument('--sets', nargs='+', default=['pref', 'green'])
    ap.add_argument('--epochs', type=int, nargs='+', default=list(EPOCHS))
    ap.add_argument('--name', default='aggregate')
    args = ap.parse_args()
    res = {}
    for pr in args.pairs:
        ex, nx = pr.split(',')
        rp = {}
        for s in args.sets:
            for q in ('L1', 'FP'):
                for T in TS:
                    pts = {}
                    for e in args.epochs:
                        ze, zr = load(args.sub, ex, e, s), load(args.sub, nx, e, s)
                        B = boot(cells(ze), len(ze['correct_sign']))
                        pts[e] = point(ze, zr, q, T, B)
                    n_pass = sum(p['pass'] for p in pts.values())
                    rp[f'{s}_{q}_T{T}'] = {'points': pts, 'n_pass': n_pass, 'pair_pass': n_pass >= 3,
                                           'n_noexp_reverse': sum(p['noexp_reverse'] for p in pts.values())}
        res[pr] = rp
    main_key = 'pref_L1_T50'
    n_pairs = len(args.pairs)
    need = {2: 2, 3: 2, 4: 3}.get(n_pairs, int(np.ceil(2 * n_pairs / 3)))
    passed = [pr for pr in args.pairs if res[pr][main_key]['pair_pass']]
    summary = {'main': main_key, 'n_pairs': n_pairs, 'needed': need, 'pairs_passed': passed,
               'gate_pass': len(passed) >= need,
               'per_pair': {pr: {'n_pass': res[pr][main_key]['n_pass'],
                                 'n_noexp_reverse': res[pr][main_key]['n_noexp_reverse'],
                                 'ba_exp': [res[pr][main_key]['points'][e]['ba_exp'] for e in args.epochs],
                                 'ba_noexp': [res[pr][main_key]['points'][e]['ba_noexp'] for e in args.epochs],
                                 'diff': [res[pr][main_key]['points'][e]['diff'] for e in args.epochs]}
                          for pr in args.pairs}}
    os.makedirs(os.path.join(BASE, args.sub), exist_ok=True)
    with open(os.path.join(BASE, args.sub, f'{args.name}.json'), 'w') as f:
        json.dump({'summary': summary, 'detail': res}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
