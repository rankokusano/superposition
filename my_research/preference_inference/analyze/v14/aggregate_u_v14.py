"""
v14-B judgement (docs/v14_experiment_log.md Sec.4, 10) from eval_u_v14.py outputs (pref test).
Point passes if: experienced sign agreement > 0.5 on BOTH layouts, AND experienced u-agreement BA has a 95% CI
(1000 episode bootstrap, RandomState(0), every cell kept) with lower bound > 0.5, AND the paired difference
(experienced - no-experience, same resampled episodes) has lower bound > 0. Pair passes with >= 3 of 5 late-5
points; 2 of 3 pairs (3 of 4 with s3). T5 is reported next to it (not a pass condition).
"""

import argparse
import json
import os

import numpy as np

from aggregate_inverse_v14 import BASE, EPOCHS, boot, ci


def ba(v, cl, idx=None):
    if idx is not None:
        v, cl = v[idx], cl[idx]
    return float(np.mean([v[cl == k].mean() for k in np.unique(cl)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--sub', default='stage2u/u')
    ap.add_argument('--name', default='aggregate_u')
    ap.add_argument('--set', default='pref')
    a = ap.parse_args()
    d = os.path.join(BASE, a.sub)
    res = {}
    for s in a.seeds:
        pts = {}
        for e in EPOCHS:
            ze, zr = np.load(os.path.join(d, f'u_mixed_s{s}_ep{e}.npz')), np.load(os.path.join(d, f'u_random_s{s}_ep{e}.npz'))
            je, jr = (json.load(open(os.path.join(d, f'u_{c}_s{s}_ep{e}.json')))[a.set] for c in ('mixed', 'random'))
            cl = ze[f'{a.set}_cell']
            ae, ar = ze[f'{a.set}_agree_ep'], zr[f'{a.set}_agree_ep']
            B = boot(cl, len(cl))
            be = np.array([ba(ae, cl, i) for i in B]); br = np.array([ba(ar, cl, i) for i in B])
            p = {'u_ba_exp': ba(ae, cl), 'u_ba_exp_ci': ci(be), 'u_ba_noexp': ba(ar, cl), 'u_ba_noexp_ci': ci(br),
                 'diff': ba(ae, cl) - ba(ar, cl), 'diff_ci': ci(be - br),
                 'sign_agree_exp': [je['sign_agree_original'], je['sign_agree_swap_rg']],
                 'sign_agree_noexp': [jr['sign_agree_original'], jr['sign_agree_swap_rg']],
                 'u_pos_by_cell_exp': je['u_pos_by_cell'], 'u_pos_by_cell_noexp': jr['u_pos_by_cell']}
            p['pass'] = bool(min(p['sign_agree_exp']) > 0.5 and p['u_ba_exp_ci'][0] > 0.5 and p['diff_ci'][0] > 0)
            pts[e] = p
        n = sum(p['pass'] for p in pts.values())
        res[f's{s}'] = {'points': pts, 'n_pass': n, 'pair_pass': n >= 3}
    need = {2: 2, 3: 2, 4: 3}[len(a.seeds)]
    passed = [k for k, v in res.items() if v['pair_pass']]
    summary = {'set': a.set, 'n_pairs': len(a.seeds), 'needed': need, 'pairs_passed': passed, 'gate_pass': len(passed) >= need,
               'per_pair': {k: {'n_pass': v['n_pass'],
                                'u_ba_exp': [round(p['u_ba_exp'], 3) for p in v['points'].values()],
                                'u_ba_noexp': [round(p['u_ba_noexp'], 3) for p in v['points'].values()],
                                'sign_agree_exp': [[round(x, 3) for x in p['sign_agree_exp']] for p in v['points'].values()]}
                            for k, v in res.items()}}
    json.dump({'summary': summary, 'detail': res}, open(os.path.join(d, a.name + '.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
