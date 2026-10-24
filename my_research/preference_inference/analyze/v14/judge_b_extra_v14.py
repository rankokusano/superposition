"""
v14-B extra conditions for writing "G writes the preference" (docs/v14_experiment_log.md Sec.30, pre-registered for
the v13 models; applied to the v12 models only as a description):
  (a) T5 of the experienced model > 0 with a 95% CI (1000 bootstrap resamples of the green-liked T5 episodes,
      RandomState(0), as in v12 Sec.28 D) above 0;
  (b) among episodes where preference and A-2's direction disagree, the share where u follows preference has a
      CI lower bound > 0.5 (u_direction_v14.py output).
Each counted over the late-5; a condition holds for a pair with >= 3 of 5 points.
Usage: python analyze/v14/judge_b_extra_v14.py --exp_cond mixed80 --dir_name u_direction_v13 --name judge_b_extra_v13
"""

import argparse
import json
import os

import numpy as np

from aggregate_inverse_v14 import BASE, EPOCHS, ci


def t5_ci(dT5, ep, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    eps = np.unique(ep)
    groups = {e: dT5[ep == e] for e in eps}
    bs = []
    for _ in range(n):
        pick = eps[rng.randint(0, len(eps), len(eps))]
        bs.append(np.concatenate([groups[e] for e in pick]).mean())
    return ci(bs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_cond', default='mixed80')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--dir_name', default='u_direction_v13')
    ap.add_argument('--name', default='judge_b_extra_v13')
    a = ap.parse_args()
    dd = json.load(open(os.path.join(BASE, 'stage2u', a.dir_name + '.json')))
    out = {}
    for s in a.seeds:
        rows = {}
        for e in EPOCHS:
            z = np.load(os.path.join(BASE, 'stage2u', 'eval', f'pref_{a.exp_cond}_s{s}_ep{e}_per.npz'))
            c = t5_ci(z['dT5'], z['t5_green_ep'])
            r = dd[f'{a.exp_cond}_s{s}|ep{e}|all']
            rows[e] = {'T5': float(z['dT5'].mean()), 'T5_ci': c, 'a': bool(c[0] > 0),
                       'follow_pref': r['disagree_follow_pref'], 'follow_pref_ci': r['disagree_follow_pref_ci'],
                       'b': bool(r['disagree_follow_pref_ci'][0] > 0.5)}
        na, nb = sum(v['a'] for v in rows.values()), sum(v['b'] for v in rows.values())
        out[f's{s}'] = {'points': rows, 'n_a': na, 'n_b': nb, 'a_holds': na >= 3, 'b_holds': nb >= 3}
        print(f"s{s}: (a) T5 {[round(v['T5'], 4) for v in rows.values()]} -> {na}/5   "
              f"(b) follow pref {[round(v['follow_pref'], 3) for v in rows.values()]} -> {nb}/5")
    json.dump(out, open(os.path.join(BASE, 'stage2u', a.name + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
