"""
v15 Sec.39 H4 judgement (fewer observations in phase 2: 100 train episodes, 1500 epochs = the same 15000 updates; pre-registered
in docs/v15_experiment_log.md Sec.39.2-39.4 before the runs). CPU evaluations of the last checkpoint (ep1500) only. Same per-pair
measures as judge_AB_v15.py (930 generated pairs, paired differences, 2000 bootstrap draws over pairs, seed 0).
    precondition: in exp or in noexp, >= 2 of 3 seeds have pair accuracy CI lower bound > 0.5 (otherwise 'floor', not judged)
    'helps': exp - noexp > 0 in all 3 seeds and the CI excludes 0 in >= 2 seeds; 'does not help': mean over seeds <= 0
Descriptive: X / Y side accuracies and their exp - noexp, G 10-step readout, m, d, the 1000-episode values next to them,
and the self-vision losses on the 100 train episodes (eval log) and on the 300 unused test episodes (test log).

    python analyze/v15/judge_H4_v15.py   # -> data/result/v15/stage2/judge_H4_v15i_cpu.json
"""

import json
import os
import sys

import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15.judge_AB_v15 import S2, boot  # noqa  (SOURCE = 'cpu')
from analyze.v15.judge_CD_v15 import RES, load  # noqa

SEEDS = (0, 1, 2)


def losses(exp, s, split):
    p = os.path.join(RES, exp + ('' if s == 0 else f'_s{s}'), str(s), 'log', split, 'self_vision.log')
    if not os.path.exists(p):
        return None
    return {int(e): float(v) for e, v in (line.split(',') for line in open(p))}


def side(p):
    return {'pairacc': boot(p['pairacc']), 'acc_X': boot(p['acc_X']), 'acc_Y': boot(p['acc_Y']),
            'm': float(p['m'].mean()), 'd': float(p['d'].mean()), 'g10': p['g10']}


def compare(E, N):
    out = {}
    for s in SEEDS:
        out[s] = {'exp': side(E[s]), 'noexp': side(N[s]),
                  'exp_minus_noexp': boot(E[s]['pairacc'] - N[s]['pairacc']),
                  'exp_minus_noexp_X': boot(E[s]['acc_X'] - N[s]['acc_X']),
                  'exp_minus_noexp_Y': boot(E[s]['acc_Y'] - N[s]['acc_Y'])}
    return out


def main():
    E, N = load('v15_s2p2h_exp_v15i_o100'), load('v15_s2p2h_noexp_v15i_o100')
    E1k, N1k = load('v15_s2p2h_exp_v15i'), load('v15_s2p2h_noexp_v15i')
    for s in SEEDS:
        for d in (N, E1k, N1k):
            assert np.array_equal(E[s]['ids'], d[s]['ids'])
    R = {'source': 'cpu', 'checkpoint': 'ep1500 (last)', 'o100': compare(E, N), 'n1000': compare(E1k, N1k)}
    H = R['o100']
    pre = {c: sum(H[s][c]['pairacc']['ci95'][0] > 0.5 for s in SEEDS) for c in ('exp', 'noexp')}
    diffs = [H[s]['exp_minus_noexp'] for s in SEEDS]
    mean = float(np.mean([d['mean'] for d in diffs]))
    helps = all(d['mean'] > 0 for d in diffs) and sum(d['ci95'][0] > 0 for d in diffs) >= 2
    R['precondition_n_seeds_ci_lower_above_half'] = pre
    R['precondition'] = bool(max(pre.values()) >= 2)
    R['mean_exp_minus_noexp'] = mean
    R['verdict'] = ('floor (not judged)' if not R['precondition'] else
                    'helps' if helps else 'does not help' if mean <= 0 else 'neither (describe)')
    # descriptive: 'helps only on the X side' (the other with the same preference as the self, rank X)
    nx = sum(H[s]['exp_minus_noexp_X']['ci95'][0] > 0 for s in SEEDS)
    ny = sum(H[s]['exp_minus_noexp_Y']['ci95'][0] > 0 for s in SEEDS)
    R['n_seeds_ci_lower_above_0'] = {'X': int(nx), 'Y': int(ny)}
    R['helps_only_X_side'] = bool(nx >= 2 and ny <= 1)
    R['losses'] = {}
    for c in ('exp', 'noexp'):
        for s in SEEDS:
            tr = losses(f'v15_s2p2h_{c}_v15i_o100', s, 'eval'); te = losses(f'v15_s2p2h_{c}_v15i_o100', s, 'test')
            tr1k = losses(f'v15_s2p2h_{c}_v15i', s, 'eval'); te1k = losses(f'v15_s2p2h_{c}_v15i', s, 'test')
            R['losses'][f'{c}_s{s}'] = {
                'o100_train100': tr, 'o100_test300': te,
                'o100_test_min_epoch': min(te, key=te.get) if te else None,
                'n1000_ep150': {'train1000': tr1k.get(150) if tr1k else None, 'test300': te1k.get(150) if te1k else None}}
    out = os.path.join(S2, 'judge_H4_v15i_cpu.json')
    json.dump(R, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps({k: R[k] for k in ('precondition_n_seeds_ci_lower_above_half', 'precondition', 'mean_exp_minus_noexp',
                                        'verdict', 'n_seeds_ci_lower_above_0', 'helps_only_X_side')}, indent=1))


if __name__ == '__main__':
    main()
