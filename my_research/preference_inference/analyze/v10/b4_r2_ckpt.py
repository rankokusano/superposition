"""
v10 B4: 4-axis position R^2 (h1/h2 -> self/other; Ridge alpha=1, in-sample,
as regression_baseline_v4.py) from a test_v10.py output on the `eval` group
(train split). With --layout_from <dataset>, also per layout (original /
swapped episodes) using that dataset's train/layout field.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import PI_DIR, git_hash  # noqa


def r2(X, Y):
    return float(r2_score(Y, Ridge(alpha=1.0).fit(X, Y).predict(X)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--layout_from', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with h5py.File(os.path.join(args.run_dir, 'save', 'saved.h5'), 'r') as f:
        g = f[f'{args.epoch:05d}/eval']
        d = {'h1': g['state/self/hidden'][()], 'h2': g['state/other/hidden'][()],
             'self': g['self_position/input'][()], 'other': g['other_position/input'][()]}
    subsets = {'all': np.ones(len(d['h1']), bool)}
    if args.layout_from:
        with h5py.File(os.path.join(PI_DIR, 'data', 'data', args.layout_from, 'data.h5'), 'r') as f:
            lay = f['train/layout'][()]
        subsets['original'] = lay == 0
        subsets['swap_rg'] = lay == 1
    res = {'dataset': args.dataset, 'split': 'train (eval group), in-sample Ridge', 'epoch': args.epoch,
           'git_commit': git_hash(), 'r2': {}}
    for name, mk in subsets.items():
        res['r2'][name] = {f'{h}_to_{p}': r2(d[h][mk].reshape(-1, 128), d[p][mk].reshape(-1, 2))
                           for h in ('h1', 'h2') for p in ('self', 'other')}
    json.dump(res, open(args.out, 'w'), indent=2)
    print(json.dumps(res['r2']))


if __name__ == '__main__':
    main()
