"""
v15 (C) / (D) judgement (pre-registered: docs/v15_experiment_log.md Sec.33.1, 33.2; CPU evaluations, Sec.30.8). CPU only.
Same per-pair measures as judge_AB_v15.py (930 generated pairs, paired differences, 2000 bootstrap draws, seed 0).
(C): rank-Y experienced (v15_s2p2h_expY_v15i*) vs rank-X experienced (v15_s2p2h_exp_v15i*), seeds 0-2.
(D): NS experienced - NS no experience (v15_s2p2h_{exp,noexp}_v15i_ns*), against the shared S_s.

    python analyze/v15/judge_CD_v15.py   # -> data/result/v15/stage2/judge_CD_v15i_cpu.json
"""

import json
import os
import sys

import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15.judge_AB_v15 import S2, boot, per_pair  # noqa  (SOURCE = 'cpu')

RES = os.path.join(_PI_DIR, 'data', 'result')
SEEDS = (0, 1, 2)
# Sec.33 constants (fixed before the runs)
X_M_MIN, W_X = 0.418, 0.064
S_BAR = -0.113
D_THR = abs(S_BAR) / 2  # 0.057 (rounded in the log)
D0_MIN = 0.60


def name(stem, s):
    return stem + ('' if s == 0 else f'_s{s}')


def load(stem):
    return {s: per_pair(name(stem, s), s) for s in SEEDS}


def summary(p):
    return {'m': float(p['m'].mean()), 'd': float(p['d'].mean()), 'acc_X': float(p['acc_X'].mean()),
            'acc_Y': float(p['acc_Y'].mean()), 'pairacc': boot(p['pairacc']), 'g10': p['g10']}


def hide_json(stem, s):
    n = name(stem, s)
    return json.load(open(os.path.join(S2, f"hide_{n if s == 0 else n + f'_seed{s}'}_cpu.json")))


def test_loss(exp, s, key='self_vision', epoch=150):
    for line in open(os.path.join(RES, exp, str(s), 'log', 'test', f'{key}.log')):
        e, v = line.split(',')
        if int(e) == epoch:
            return float(v)


def judge_C(R):
    X, Y, N = load('v15_s2p2h_exp_v15i'), load('v15_s2p2h_expY_v15i'), load('v15_s2p2h_noexp_v15i')
    for s in SEEDS:
        assert np.array_equal(X[s]['ids'], Y[s]['ids']) and np.array_equal(X[s]['ids'], N[s]['ids'])
    C = {'Y': {s: summary(Y[s]) for s in SEEDS}, 'X': {s: summary(X[s]) for s in SEEDS}}
    for s in SEEDS:
        C['Y'][s].update({'dm': boot(Y[s]['m'] - X[s]['m']), 'dd': boot(Y[s]['d'] - X[s]['d']),
                          'accX_change': boot(Y[s]['acc_X'] - X[s]['acc_X']), 'accY_change': boot(Y[s]['acc_Y'] - X[s]['acc_Y'])})
    mY = [C['Y'][s]['m'] for s in SEEDS]; mX = [C['X'][s]['m'] for s in SEEDS]
    dY = [C['Y'][s]['d'] for s in SEEDS]; dX = [C['X'][s]['d'] for s in SEEDS]; dN = [float(N[s]['d'].mean()) for s in SEEDS]
    assert abs(min(mX) - X_M_MIN) < 5e-4 and abs(max(mX) - min(mX) - W_X) < 1e-3, (mX,)
    dm_bar = float(np.mean([C['Y'][s]['dm']['mean'] for s in SEEDS]))
    c1 = max(mY) < min(mX)
    c2 = all(C['Y'][s]['dm']['ci95'][1] < 0 for s in SEEDS)
    c3 = not (max(mY) < min(mX) or min(mY) > max(mX))
    c4 = abs(dm_bar) < W_X / 2
    C.update({'m_Y_range': [min(mY), max(mY)], 'm_X_range': [min(mX), max(mX)], 'dm_bar': dm_bar,
              'd_Y_range': [min(dY), max(dY)], 'd_X_range': [min(dX), max(dX)], 'd_noexp_range': [min(dN), max(dN)],
              'dd_bar': float(np.mean([C['Y'][s]['dd']['mean'] for s in SEEDS])),
              'C1': bool(c1), 'C2': bool(c2), 'C3': bool(c3), 'C4': bool(c4),
              'verdict': 'separated' if c1 and c2 else 'not separated' if c3 and c4 else 'neither (describe)'})
    R['C'] = C


def judge_D(R):
    SE, SN = load('v15_s2p2h_exp_v15i'), load('v15_s2p2h_noexp_v15i')
    NE, NN = load('v15_s2p2h_exp_v15i_ns'), load('v15_s2p2h_noexp_v15i_ns')
    for s in SEEDS:
        for d in (SN, NE, NN):
            assert np.array_equal(SE[s]['ids'], d[s]['ids'])
    D = {}
    for s in SEEDS:
        D[s] = {'N': boot(NE[s]['pairacc'] - NN[s]['pairacc']), 'S': boot(SE[s]['pairacc'] - SN[s]['pairacc']),
                'N_minus_S': boot((NE[s]['pairacc'] - NN[s]['pairacc']) - (SE[s]['pairacc'] - SN[s]['pairacc'])),
                'ns_exp': summary(NE[s]), 'ns_noexp': summary(NN[s]),
                'D0_readable': bool(NN[s]['pairacc'].mean() >= D0_MIN)}
        for c, stem in (('exp', 'v15_s2p2h_exp_v15i_ns'), ('noexp', 'v15_s2p2h_noexp_v15i_ns')):
            h = hide_json(stem, s)
            D[s][f'ns_{c}']['g_curve'] = [h['gen']['memory_readout']['g'][str(o)]['mean'] for o in range(11)]
            D[s][f'ns_{c}']['ctrl3_max_h2_diff'] = h['ctrl3']['max_h2_diff_upto_te_plus_K']
            D[s][f'ns_{c}']['gen_max_h2_diff_upto_te'] = h['gen']['max_h2_diff_upto_te']
            D[s][f'ns_{c}']['dropped'] = h['gen']['dropped_pairs_input_differs_before_te_plus_K']
            D[s][f'phase1_test_self_vision_{c}'] = {'ns': test_loss(f'v15_s2p1_{c}_v15i_ns_mse', s), 'shared': test_loss(f'v15_s2p1_{c}_v15i_mse', s)}
    readable = [s for s in SEEDS if D[s]['D0_readable']]
    D['readable_seeds'] = readable
    D['N_bar'] = float(np.mean([D[s]['N']['mean'] for s in readable])) if readable else None
    D['S_bar_recomputed'] = float(np.mean([D[s]['S']['mean'] for s in SEEDS]))
    neg = sum(1 for s in readable if D[s]['N']['ci95'][1] < 0)
    D['n_seeds_N_ci_upper_below_0'] = neg
    D['memory_kept_all_6'] = bool(all(D[s][f'ns_{c}']['g10']['ci95'][0] > 0.5 for s in SEEDS for c in ('exp', 'noexp')))
    if len(readable) < 2:
        D['verdict'] = 'not judged (D0: fewer than 2 readable seeds)'
    else:
        nb = D['N_bar']
        D['D1'] = bool(abs(nb) < D_THR); D['D2'] = bool(neg <= 1)
        D['D3'] = bool(nb <= -D_THR); D['D4'] = bool(neg >= 2)
        D['verdict'] = ('shared pathway' if D['D1'] and D['D2'] else
                        'unrelated to sharing' if D['D3'] and D['D4'] else 'neither (describe)')
    R['D'] = D


def main():
    R = {'source': 'cpu'}
    judge_C(R)
    judge_D(R)
    out = os.path.join(S2, 'judge_CD_v15i_cpu.json')
    json.dump(R, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


if __name__ == '__main__':
    main()
