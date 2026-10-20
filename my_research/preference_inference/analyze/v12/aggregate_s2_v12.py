"""
v12 stage-2 primary judgement (docs/v12_experiment_log.md Sec.28): runs v11's aggregate_s2_v11 unchanged on the v12
outputs (data/result/v12/<out>/eval) and writes aggregate_primary_<tag>.json there (the v11 file name is renamed so
interim per-pair aggregations do not overwrite each other).
Usage: python analyze/v12/aggregate_s2_v12.py --seeds 0 1 2 [--out stage2] [--epochs 160 ... 200] [--tag s0s1s2]
Note: v11's rule inside is "2 pairs -> both, otherwise >= 2"; with 4 pairs (s3 added) the v12 rule is >= 3 (Sec.26),
which is applied here on top of the per-pair results.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES  # noqa
import aggregate_s2_v11 as agg  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--out', default='stage2')
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    agg.EV = os.path.join(RES, 'v12', a.out, 'eval')
    agg.EPS = a.epochs
    sys.argv = [agg.__file__, '--seeds'] + [str(s) for s in a.seeds]
    agg.main()
    src = os.path.join(agg.EV, 'aggregate_primary.json')
    res = json.load(open(src))
    n = len(a.seeds)
    need = {1: 1, 2: 2, 3: 2, 4: 3}[n]    # Sec.26 (1 pair: interim only)
    for k, v in res.items():
        if ':' in k:
            v['setting_pass_v12_rule'] = sum(v[f's{s}']['pair_pass'] for s in a.seeds) >= need
            v['n_pairs'] = n
    res['note'] = 'interim (single pair)' if n == 1 else f'v12 rule Sec.26: {need} of {n}'
    tag = a.tag or 's' + ''.join(str(s) for s in a.seeds)
    json.dump(res, open(os.path.join(agg.EV, f'aggregate_primary_{tag}.json'), 'w'), indent=1)
    os.remove(src)
    print({k: v['setting_pass_v12_rule'] for k, v in res.items() if ':' in k})


if __name__ == '__main__':
    main()
