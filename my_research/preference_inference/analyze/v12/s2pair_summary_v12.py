"""
v12 stage-2 interim summary for one pair (docs/v12_experiment_log.md Sec.28, 31): late-5 means of the primary and
secondary measures from the per-checkpoint outputs under data/result/v12/<out>/{eval,t1t3,t4t6,m1d_other}, for
both settings and conditions. Writes <out>/summary/pair_s<seed>.json and prints a compact table. Missing parts are
skipped (the evaluation runs in stages).
"""

import argparse
import json
import os
import statistics as st
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES  # noqa


def flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f'{pre}{k}/'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f'{pre}{k}'] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', default='stage2')
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    a = ap.parse_args()
    base = os.path.join(RES, 'v12', a.out)
    res = {}
    for part in ('eval', 't1t3', 't4t6', 'm1d_other'):
        for stg in ('green', 'pref'):
            for c in ('random', 'mixed'):
                fs = [os.path.join(base, part, f'{stg}_{c}_s{a.seed}_ep{e}.json') for e in a.epochs]
                if not all(os.path.exists(f) for f in fs):
                    continue
                F = [flat(json.load(open(f))) for f in fs]
                keys = [k for k in F[0] if all(k in f for f in F)]
                res[f'{part}/{stg}_{c}'] = {k: st.mean(f[k] for f in F) for k in keys}
    os.makedirs(os.path.join(base, 'summary'), exist_ok=True)
    json.dump(res, open(os.path.join(base, 'summary', f'pair_s{a.seed}.json'), 'w'), indent=1)
    show = [('eval', 'T2_mean/t50'), ('eval', 'T5_mean'), ('eval', 'corr_Q2_PA2/green_set/all'), ('eval', 'corr_Q2_PA2/green_set/original'),
            ('eval', 'corr_Q2_PA2/green_set/swap_rg'), ('eval', 'corr_Q2_PA2/pref_test/all'),
            ('eval', 'motion_readout/green_set/Q2_to_A1_step_dir'), ('eval', 'motion_readout/green_set/Q2_to_A2_step_dir'),
            ('t1t3', 'green/gen/self_L1'), ('t1t3', 'green/zero/self_L1'), ('t1t3', 'green/d6/self_L1'),
            ('t1t3', 'green/gen/other_fp_mse'), ('t1t3', 'green/zero/other_fp_mse'), ('t1t3', 'green/d6/other_fp_mse'),
            ('t1t3', 'green/spread/var_share_bins2_5'), ('t1t3', 'green/spread/std_all'), ('t1t3', 'green/T1_A2pos/all'),
            ('t4t6', 'gen/tracking/green/median'), ('t4t6', 'gen/T4/swap_minus_noswap'),
            ('m1d_other', 'green_gen/c_cf_other'), ('m1d_other', 'red_gen/c_cf_other')]
    for part, k in show:
        cells = []
        for stg in ('green', 'pref'):
            for c in ('random', 'mixed'):
                v = res.get(f'{part}/{stg}_{c}', {}).get(k)
                cells.append('-' if v is None else f'{v:+.4f}')
        print(f'{part:9s} {k:45s} ' + ' | '.join(cells))


if __name__ == '__main__':
    main()
