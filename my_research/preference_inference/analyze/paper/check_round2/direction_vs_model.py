"""
Paper round 2 (paper/check/paper_round2_log.md Sec.4; NOT pre-registered, saved outputs only, CPU):
difference between a model's balanced accuracy (experiment 4 = v14-A, pref test, T = 50) and the direction-only
baseline (paper round 1 Sec.1), with a paired episode bootstrap (1000, RandomState(0), every cell present; the same
resampled episodes for model and baseline). Per checkpoint: diff and 95% CI; per pair: points with CI low > 0;
across pairs: mean of the per-pair late-5 mean diffs with a t 95% CI.
Writes paper/check/direction_vs_model.json.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _PI)
for d in ('analyze/v11', 'analyze/v14', 'analyze/paper'):
    sys.path.insert(0, os.path.join(_PI, d))
from aggregate_inverse_v14 import BASE, EPOCHS, ba, boot, cells, choice, ci  # noqa
from direction_baseline_paper import direction  # noqa
from eval_s2_v11 import load_set  # noqa

T = 50
OUT = os.path.join(_PI, 'paper', 'check', 'direction_vs_model.json')
SETS = {
    'FP': {'80': ['v13mse_mixed_{}', range(6)], '50': ['v12mse_mixed_{}', (0, 1, 2, 4, 5)],
           'none': ['v12mse_random_{}', range(6)], 'ns': ['v13nsmse_mixed_{}', range(3)]},
    'L1': {'80': ['v13_mixed_{}', range(6)], '50': ['v12_mixed_{}', range(6)],
           'none': ['v12_random_{}', range(6)], 'ns': ['v13ns_mixed_{}', range(3)]},
}


def main():
    d = load_set('pref')
    dirn = direction(d['other_position'], T)
    out = {'definition': 'paper_round2_log.md Sec.4 (not pre-registered)', 'T': T, 'results': {}, 'across': {}}
    for q, conds in SETS.items():
        for cn, (pat, seeds) in conds.items():
            per_pair = {}
            for s in seeds:
                pts = {}
                for e in EPOCHS:
                    z = np.load(os.path.join(BASE, 'inverse', f'{pat.format(s)}_ep{e}_pref.npz'))
                    cl, pref = cells(z), z['correct_sign']
                    cm = np.where(pref > 0, choice(z, q, T), 1 - choice(z, q, T))
                    cd = (dirn == pref).astype(float)
                    B = boot(cl, len(pref))
                    bm = np.array([ba(cm, cl, b) for b in B])
                    bd = np.array([ba(cd, cl, b) for b in B])
                    pts[e] = {'ba_model': ba(cm, cl), 'ba_dir': ba(cd, cl), 'diff': ba(cm, cl) - ba(cd, cl),
                              'diff_ci': ci(bm - bd), 'ba_model_ci': ci(bm)}
                diffs = [p['diff'] for p in pts.values()]
                per_pair[f's{s}'] = {
                    'points': pts, 'diff_late5_mean': float(np.mean(diffs)), 'diff_min': min(diffs),
                    'diff_max': max(diffs), 'ba_model_min': min(p['ba_model'] for p in pts.values()),
                    'ba_model_max': max(p['ba_model'] for p in pts.values()),
                    'n_ci_low_above0': int(sum(p['diff_ci'][0] > 0 for p in pts.values())),
                    'n_ci_high_below0': int(sum(p['diff_ci'][1] < 0 for p in pts.values()))}
            out['results'][f'{q}|{cn}'] = per_pair
            v = np.array([r['diff_late5_mean'] for r in per_pair.values()])
            n = len(v)
            h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
            out['across'][f'{q}|{cn}'] = {'n_pairs': n, 'df': n - 1, 'pairs': list(per_pair), 'mean': float(v.mean()),
                                          'ci95': [float(v.mean() - h), float(v.mean() + h)],
                                          'n_pairs_positive': int((v > 0).sum())}
    json.dump(out, open(OUT, 'w'), indent=1)
    for k, pp in out['results'].items():
        a = out['across'][k]
        print(f"== {k}: across {a['n_pairs']} pairs mean {a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] "
              f"(df {a['df']}, positive {a['n_pairs_positive']})")
        for s, r in pp.items():
            print(f"   {s}: BA {r['ba_model_min']:.3f}-{r['ba_model_max']:.3f}  diff {r['diff_min']:+.3f}~{r['diff_max']:+.3f} "
                  f"(mean {r['diff_late5_mean']:+.3f})  CI>0 {r['n_ci_low_above0']}/5  CI<0 {r['n_ci_high_below0']}/5")


if __name__ == '__main__':
    main()
