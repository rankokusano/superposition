"""
v10 stage A, item 4-5 (docs/v10_experiment_log.md Sec.5.2, 5.9): 4-axis
position R^2 and self-vision prediction error for one checkpoint, A0
(r2_a1random_a2rl, original layout) vs A1 (v10_swap_r2, swapped layout,
same episode seeds -> A-1's path identical episode by episode).

Regressors (Ridge alpha=1, as analyze/regression_baseline_v4.py):
  (i)  refit on each condition's train split (`eval` group, in-sample R^2)
       -- "is the information there"
  (ii) fit on A0's train split, applied unchanged to A1's train split
       -- "is it the same code"
  held-out reference: fit on A0 train, scored on A0 test and A1 test.
Self-vision error = the eval loop's logged L1 loss (log/{eval,test}/self_vision.log).

Inputs: the two test_v10.py output dirs (each with save/saved.h5 holding
eval+test groups and log/). Output: one JSON.
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
from common_v10 import git_hash  # noqa

AXES = [('h1', 'self'), ('h1', 'other'), ('h2', 'self'), ('h2', 'other')]


def load(run_dir, epoch, mode):
    with h5py.File(os.path.join(run_dir, 'save', 'saved.h5'), 'r') as f:
        k = f'{epoch:05d}/{mode}'
        d = {
            'h1': f[f'{k}/state/self/hidden'][()],
            'h2': f[f'{k}/state/other/hidden'][()],
            'self': f[f'{k}/self_position/input'][()],
            'other': f[f'{k}/other_position/input'][()],
        }
    n, t, _ = d['h1'].shape
    return {key: v.reshape(n * t, -1) for key, v in d.items()}, n, t


def read_log(run_dir, mode, name):
    p = os.path.join(run_dir, 'log', mode, f'{name}.log')
    if not os.path.exists(p):
        return None
    vals = [float(line.split(',')[1]) for line in open(p) if line.strip()]
    return vals[-1]


def fit(X, Y):
    return Ridge(alpha=1.0).fit(X, Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a0_dir', required=True)
    ap.add_argument('--a1_dir', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--exp_config', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--eval_seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    a0_tr, n0, t0 = load(args.a0_dir, args.epoch, 'eval')
    a1_tr, n1, t1 = load(args.a1_dir, args.epoch, 'eval')
    a0_te, _, _ = load(args.a0_dir, args.epoch, 'test')
    a1_te, _, _ = load(args.a1_dir, args.epoch, 'test')
    assert np.array_equal(a0_tr['self'], a1_tr['self']), 'A-1 paths must pair up episode by episode'

    res = {'refit_in_sample': {'A0': {}, 'A1': {}}, 'A0_regressor_on_A1_train': {},
           'heldout_A0_regressor': {'A0_test': {}, 'A1_test': {}}}
    for h, target in AXES:
        name = f'{h}_to_{target}'
        for cond, d in (('A0', a0_tr), ('A1', a1_tr)):
            reg = fit(d[h], d[target])
            res['refit_in_sample'][cond][name] = float(r2_score(d[target], reg.predict(d[h])))
        reg0 = fit(a0_tr[h], a0_tr[target])
        res['A0_regressor_on_A1_train'][name] = float(r2_score(a1_tr[target], reg0.predict(a1_tr[h])))
        res['heldout_A0_regressor']['A0_test'][name] = float(r2_score(a0_te[target], reg0.predict(a0_te[h])))
        res['heldout_A0_regressor']['A1_test'][name] = float(r2_score(a1_te[target], reg0.predict(a1_te[h])))

    res['self_vision_loss'] = {
        cond: {mode: read_log(d, mode, 'self_vision') for mode in ('eval', 'test')}
        for cond, d in (('A0', args.a0_dir), ('A1', args.a1_dir))}
    res['feature_prediction_loss'] = {
        cond: {f'{mode}/{nm}': read_log(d, mode, nm) for mode in ('eval', 'test')
               for nm in ('feature_prediction_self', 'feature_prediction_other')}
        for cond, d in (('A0', args.a0_dir), ('A1', args.a1_dir))}
    res['meta'] = {
        'exp_config': args.exp_config, 'seed': args.seed, 'epoch': args.epoch,
        'eval_seed': args.eval_seed, 'mask': 'on (config p_mask_vision), cudnn deterministic',
        'A0_dataset': 'r2_a1random_a2rl', 'A1_dataset': 'v10_swap_r2',
        'train_split': 'eval group (train split, 3000 ep, unshuffled)', 'test_split': 'test group (300 ep)',
        'n_train_episodes': int(n0), 'n_steps': int(t0), 'git_commit': git_hash(),
        'a0_dir': args.a0_dir, 'a1_dir': args.a1_dir,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
