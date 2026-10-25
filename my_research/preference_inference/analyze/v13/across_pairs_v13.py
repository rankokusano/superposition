"""
Across-pair statistics (docs/v13_experiment_log.md Sec.33; outside the pre-registration, descriptive, CPU).
Each pair contributes one value per model: the late-5 mean of the per-episode values (mean over the 5 checkpoints of
the episode means). Paired comparisons across pairs: 80% - 50%, 80% - none, 50% - none for G3, G5b and the
sign-following part of G5a (sets (1), (2)). For all 6 pairs (s0-s5) and the 3 main pairs (s0-s2): mean difference,
95% CI with the t distribution (df = n - 1), two-sided exact sign test, number of positive pairs.
Writes data/result/v13/extra/across_pairs.json; prints a markdown table.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES  # noqa

EPS = (160, 170, 180, 190, 200)
SEEDS = (0, 1, 2, 3, 4, 5)
N = lambda *a: np.load(os.path.join(RES, *a))  # noqa


def s1(cond, s):
    if cond == '80':
        return ('v13', 'main', 's1eval', f's{s}')
    return ('v12', 'main', 's1eval', f's{s}') if s <= 3 else ('v13', 'v12ref', 's1eval', f's{s}')


def pair_values(s):
    v = {}
    for e in EPS:
        p80, p50 = N(*s1('80', s), f'per_ep{e}.npz'), N(*s1('50', s), f'per_ep{e}.npz')
        g80, g50 = N('v13', 'main', 'g5', f's{s}', f'g5_per_ep{e}.npz'), N('v13', 'v12ref', 'g5', f's{s}', f'g5_per_ep{e}.npz')
        d = {c: N('v13', 'post', 'g5a_decomp', f's{s}', f'decomp_{c}_ep{e}.npz') for c in ('random', 'mixed50', 'mixed80')}
        add = lambda q, c, x: v.setdefault(q, {}).setdefault(c, []).append(float(np.nanmean(x)))  # noqa
        add('G3', 'none', p80['random__freeze_K15_hold']); add('G3', '50', p50['mixed__freeze_K15_hold']); add('G3', '80', p80['mixed__freeze_K15_hold'])
        add('G5b', 'none', g80['random__G5b']); add('G5b', '50', g50['mixed__G5b']); add('G5b', '80', g80['mixed__G5b'])
        for k in ('green', 'pref'):
            add(f'sign_{k}', 'none', (g80[f'random__G5a_{k}_K15_freeze'] - d['random'][f'{k}_flip_g']) / 2)
            add(f'sign_{k}', '50', (g50[f'mixed__G5a_{k}_K15_freeze'] - d['mixed50'][f'{k}_flip_g']) / 2)
            add(f'sign_{k}', '80', (g80[f'mixed__G5a_{k}_K15_freeze'] - d['mixed80'][f'{k}_flip_g']) / 2)
    return {q: {c: float(np.mean(x)) for c, x in cs.items()} for q, cs in v.items()}


def test(diff):
    diff = np.asarray(diff, float)
    n = len(diff)
    m, sd = diff.mean(), diff.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    npos, nz = int((diff > 0).sum()), int((diff != 0).sum())
    p = float(min(1.0, 2 * min(stats.binom.cdf(npos, nz, 0.5), stats.binom.sf(npos - 1, nz, 0.5))))
    return {'mean': float(m), 'ci95': [float(m - h), float(m + h)], 'sign_p': p, 'n_pos': npos, 'n': n}


def main():
    vals = {s: pair_values(s) for s in SEEDS}
    out = {'per_pair': {f's{s}': vals[s] for s in SEEDS}, 'tests': {}}
    names = {'G3': 'G3', 'G5b': 'G5b', 'sign_green': '符号に従う分 ①', 'sign_pref': '符号に従う分 ②'}
    comps = (('80', '50', '80% − 50%'), ('80', 'none', '80% − 経験なし'), ('50', 'none', '50% − 経験なし'))
    print('| 項目 | 比較 | 全 6 組：差の平均 [95% CI] | 符号検定 p | 正の組 | 本番 3 組：差の平均 [95% CI] | 符号検定 p | 正の組 |')
    print('|---|---|---|---|---|---|---|---|')
    for q, nm in names.items():
        for a, b, lab in comps:
            diff = [vals[s][q][a] - vals[s][q][b] for s in SEEDS]
            t6, t3 = test(diff), test(diff[:3])
            out['tests'][f'{q}_{a}_minus_{b}'] = {'all6': t6, 'main3': t3, 'per_pair_diff': diff}
            f = lambda t: f"{t['mean']:+.3f} [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | {t['sign_p']:.3f} | {t['n_pos']}/{t['n']}"  # noqa
            print(f'| {nm} | {lab} | {f(t6)} | {f(t3)} |')
    os.makedirs(os.path.join(RES, 'v13', 'extra'), exist_ok=True)
    json.dump(out, open(os.path.join(RES, 'v13', 'extra', 'across_pairs.json'), 'w'), indent=1)
    print('\n組ごとの late-5 の平均（経験なし / 50% / 80%）：')
    for s in SEEDS:
        print(f's{s}: ' + '；'.join(f"{names[q]} {vals[s][q]['none']:+.3f} / {vals[s][q]['50']:+.3f} / {vals[s][q]['80']:+.3f}" for q in names))


if __name__ == '__main__':
    main()
