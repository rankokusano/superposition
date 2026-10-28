"""
v15 Sec.38 judgement of the last check "reset only the other side's 11 input columns" (pre-registered: docs/v15_experiment_log.md
Sec.38.4-38.6, constants fixed before the runs from the CPU values of the existing runs). CPU evaluations only.
Same per-pair measures as judge_AB_v15.py / judge_CD_v15.py (930 generated pairs, paired differences, 2000 bootstrap
draws over pairs, seed 0).
    R1 (main) v15_s2p2h_noexp_v15i_r1*: R1 - shared noexp (same seed). "pathway needed" / "not needed" / describe.
    R2 (secondary, descriptive) v15_s2p2h_noexp_v15i_r2*: R2 - NS noexp (same seed).
    G's memory (descriptive): G 10-step readout of R1 next to the shared and NS values.
    R1b (Sec.39.1, only if run) v15_s2p2h_noexp_v15i_r1b*: R1 with the 11 columns scaled to the shared norm; same rule as R1,
    and the reading of the R1 / R1b pair.
    R2b (Sec.41, only if run) v15_s2p2h_noexp_v15i_r2b*: NS with the 11 columns scaled to the shared norm; share of the R2 rise
    kept, mean(R2b - NS) / mean(R2 - NS) over seeds: < 1/2 -> 'learned content works', >= 1/2 -> 'entrance size works'.

    python analyze/v15/judge_R12_v15.py   # -> data/result/v15/stage2/judge_R12_v15i_cpu.json
"""

import json
import os
import sys

import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15.judge_AB_v15 import S2, boot  # noqa  (SOURCE = 'cpu')
from analyze.v15.judge_CD_v15 import hide_json, load  # noqa

SEEDS = (0, 1, 2)
# Sec.38.4 constants (fixed before the runs): mean over seeds of shared noexp - NS noexp pair accuracy (CPU)
GAP_SN = 0.3197
NEED_MIN = GAP_SN / 2      # 0.160
NOT_NEEDED_MAX = GAP_SN / 4  # 0.080
# Sec.38.6 (descriptive): G 10-step readout, mean over seeds (CPU)
G10_SH, G10_NS = 0.8690, 0.5901
G10_KEPT = G10_SH - (G10_SH - G10_NS) / 4   # 0.799
G10_LOST = G10_SH - (G10_SH - G10_NS) / 2   # 0.730


def summ(p, stem, s):
    h = hide_json(stem, s)
    return {'pairacc': boot(p['pairacc']), 'acc_X': float(p['acc_X'].mean()), 'acc_Y': float(p['acc_Y'].mean()),
            'm': float(p['m'].mean()), 'd': float(p['d'].mean()), 'g10': p['g10'],
            'g_curve': [h['gen']['memory_readout']['g'][str(o)]['mean'] for o in range(11)],
            'h2_10': h['gen']['memory_readout']['h2']['10'],
            'dropped': h['gen']['dropped_pairs_input_differs_before_te_plus_K'],
            'ctrl3_max_h2_diff': h['ctrl3']['max_h2_diff_upto_te_plus_K']}


def main():
    stems = {'shared': 'v15_s2p2h_noexp_v15i', 'ns': 'v15_s2p2h_noexp_v15i_ns',
             'r1': 'v15_s2p2h_noexp_v15i_r1', 'r2': 'v15_s2p2h_noexp_v15i_r2'}
    P = {k: load(v) for k, v in stems.items()}
    for s in SEEDS:
        for k in P:
            assert np.array_equal(P[k][s]['ids'], P['shared'][s]['ids'])
    gap = float(np.mean([P['shared'][s]['pairacc'].mean() - P['ns'][s]['pairacc'].mean() for s in SEEDS]))
    assert abs(gap - GAP_SN) < 5e-4, gap
    R = {'source': 'cpu', 'gap_shared_minus_ns': gap, 'need_min': NEED_MIN, 'not_needed_max': NOT_NEEDED_MAX,
         'g10_kept_min': G10_KEPT, 'g10_lost_max': G10_LOST, 'per_seed': {}}
    for s in SEEDS:
        R['per_seed'][s] = {k: summ(P[k][s], stems[k], s) for k in P}
        R['per_seed'][s].update({
            'r1_minus_shared': boot(P['r1'][s]['pairacc'] - P['shared'][s]['pairacc']),
            'r1_minus_ns': boot(P['r1'][s]['pairacc'] - P['ns'][s]['pairacc']),
            'r2_minus_ns': boot(P['r2'][s]['pairacc'] - P['ns'][s]['pairacc']),
            'r2_minus_shared': boot(P['r2'][s]['pairacc'] - P['shared'][s]['pairacc'])})
    drop = -float(np.mean([R['per_seed'][s]['r1_minus_shared']['mean'] for s in SEEDS]))
    all_below = all(R['per_seed'][s]['r1_minus_shared']['ci95'][1] < 0 for s in SEEDS)
    R['R1'] = {'mean_drop': drop, 'all_seeds_ci_upper_below_0': bool(all_below),
               'verdict': ('pathway needed' if all_below and drop >= NEED_MIN else
                           'not needed' if drop < NOT_NEEDED_MAX else 'neither (describe)')}
    R['R2'] = {'mean_rise_over_ns': float(np.mean([R['per_seed'][s]['r2_minus_ns']['mean'] for s in SEEDS]))}
    if all(os.path.exists(os.path.join(S2, f"hide_v15_s2p2h_noexp_v15i_r1b{'' if s == 0 else f'_s{s}_seed{s}'}_cpu.json"))
           for s in SEEDS):
        Pb = load('v15_s2p2h_noexp_v15i_r1b')
        rb = {s: {'r1b': summ(Pb[s], 'v15_s2p2h_noexp_v15i_r1b', s),
                  'r1b_minus_shared': boot(Pb[s]['pairacc'] - P['shared'][s]['pairacc']),
                  'r1b_minus_r1': boot(Pb[s]['pairacc'] - P['r1'][s]['pairacc'])} for s in SEEDS}
        drop_b = -float(np.mean([rb[s]['r1b_minus_shared']['mean'] for s in SEEDS]))
        below_b = all(rb[s]['r1b_minus_shared']['ci95'][1] < 0 for s in SEEDS)
        vb = ('pathway needed' if below_b and drop_b >= NEED_MIN else
              'not needed' if drop_b < NOT_NEEDED_MAX else 'neither (describe)')
        v1 = R['R1']['verdict']
        R['R1b'] = {'per_seed': rb, 'mean_drop': drop_b, 'all_seeds_ci_upper_below_0': bool(below_b), 'verdict': vb,
                    'reading': ('learned content needed' if v1 == 'pathway needed' and vb == 'pathway needed' else
                                'entrance size needed (content unknown)' if v1 == 'pathway needed' and vb == 'not needed'
                                else 'describe')}
    g10 = float(np.mean([R['per_seed'][s]['r1']['g10']['mean'] for s in SEEDS]))
    R['G_memory'] = {'r1_g10_mean': g10, 'reading': ('kept' if g10 >= G10_KEPT else 'lost' if g10 <= G10_LOST else 'between'),
                     'r2_g10_mean': float(np.mean([R['per_seed'][s]['r2']['g10']['mean'] for s in SEEDS]))}
    if all(os.path.exists(os.path.join(S2, f"hide_v15_s2p2h_noexp_v15i_r2b{'' if s == 0 else f'_s{s}_seed{s}'}_cpu.json"))
           for s in SEEDS):
        Pc = load('v15_s2p2h_noexp_v15i_r2b')
        rc = {s: {'r2b': summ(Pc[s], 'v15_s2p2h_noexp_v15i_r2b', s),
                  'r2b_minus_ns': boot(Pc[s]['pairacc'] - P['ns'][s]['pairacc']),
                  'r2b_minus_shared': boot(Pc[s]['pairacc'] - P['shared'][s]['pairacc']),
                  'r2b_minus_r2': boot(Pc[s]['pairacc'] - P['r2'][s]['pairacc'])} for s in SEEDS}
        for s in SEEDS:
            rc[s]['share_of_r2_rise'] = rc[s]['r2b_minus_ns']['mean'] / R['per_seed'][s]['r2_minus_ns']['mean']
        rise_b = float(np.mean([rc[s]['r2b_minus_ns']['mean'] for s in SEEDS]))
        share = rise_b / R['R2']['mean_rise_over_ns']
        R['R2b'] = {'per_seed': rc, 'mean_rise_over_ns': rise_b, 'share_of_r2_rise': share,
                    'reading': 'learned content works' if share < 0.5 else 'entrance size works (content unknown)'}
    out = os.path.join(S2, 'judge_R12_v15i_cpu.json')
    json.dump(R, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'per_seed'} if isinstance(v, dict) else v)
                      for k, v in R.items() if k != 'per_seed'}, indent=1))


if __name__ == '__main__':
    main()
