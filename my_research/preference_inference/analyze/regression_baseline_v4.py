"""
v4 §5.0.2: baseline re-measurement.

Applies the SAME 4-axis Ridge R^2 methodology as analyze/regression_v3.py
to an arbitrary saved.h5 (root-repo exp1_l1 / exp3, or preference_inference's
own v3 results), so all baseline numbers in the v4 instruction doc's §2.1
table are backed by a file this script wrote, not by memory of a past
session.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/regression_baseline_v4.py \
        --saved_h5 /work/data/result/exp1_l1/0/test/self_random_other_stay/save/saved.h5 \
        --epoch 200 --mode eval --label exp1_l1

    python analyze/regression_baseline_v4.py \
        --saved_h5 /work/data/result/exp3/0/test/self_random_other_stay_periodic/save/saved.h5 \
        --epoch 200 --mode eval --label exp3

    python analyze/regression_baseline_v4.py \
        --saved_h5 /work/my_research/preference_inference/data/result/v3_exp_b_mgve/0/test/v3_b_mgve_train/save/saved.h5 \
        --epoch 200 --mode eval --label v3_b_mgve
"""

import argparse
import json
import os
import sys

import h5py
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import util  # noqa

SAVE_DIR = 'data/result/baseline_v4'


def load_data(saved_h5, epoch, mode):
    key_pre = f'{epoch:05d}/{mode}'
    with h5py.File(saved_h5, 'r') as f:
        h1 = f[f'{key_pre}/state/self/hidden'][()]
        h2 = f[f'{key_pre}/state/other/hidden'][()]
        sp = f[f'{key_pre}/self_position/input'][()]
        op = f[f'{key_pre}/other_position/input'][()]
    N, T, D = h1.shape
    h1 = h1.reshape(N * T, D)
    h2 = h2.reshape(N * T, D)
    sp = sp.reshape(N * T, 2)
    op = op.reshape(N * T, 2)
    return h1, h2, sp, op, N, T, D


def r2(X, Y):
    reg = Ridge(alpha=1.0)
    reg.fit(X, Y)
    pred = reg.predict(X)
    return r2_score(Y, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_h5', required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--label', required=True)
    parser.add_argument('--exp_config', default=None,
                         help='name of the config/exp/<name>.yml this saved_h5 came from '
                              '(recorded verbatim, incl. full yaml content, in the output JSON)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataset_name', default=None,
                         help='dataset the model was tested on (test_data_name passed to test.py)')
    args = parser.parse_args()

    # best-effort fallback if the caller didn't pass --exp_config/--seed/
    # --dataset_name explicitly: saved_h5 paths follow
    # data/result/<exp_config>/<seed>/test/<dataset_name>/save/saved.h5
    exp_config_name, seed, dataset_name = args.exp_config, args.seed, args.dataset_name
    if exp_config_name is None or seed is None or dataset_name is None:
        parts = args.saved_h5.replace('\\', '/').split('/')
        if 'result' in parts and 'test' in parts:
            ri = parts.index('result')
            ti = parts.index('test')
            if exp_config_name is None and ri + 1 < len(parts):
                exp_config_name = parts[ri + 1]
            if seed is None and ri + 2 < len(parts):
                try:
                    seed = int(parts[ri + 2])
                except ValueError:
                    pass
            if dataset_name is None and ti + 1 < len(parts):
                dataset_name = parts[ti + 1]

    if not os.path.exists(args.saved_h5):
        raise FileNotFoundError(args.saved_h5)

    os.makedirs(SAVE_DIR, exist_ok=True)

    h1, h2, sp, op, N, T, D = load_data(args.saved_h5, args.epoch, args.mode)
    print(f'[{args.label}] source={args.saved_h5} epoch={args.epoch} mode={args.mode} '
          f'N={N} T={T} D={D} n_samples={N * T}')

    r2_h1_self = r2(h1, sp)
    r2_h1_other = r2(h1, op)
    r2_h2_self = r2(h2, sp)
    r2_h2_other = r2(h2, op)

    result = {
        'label': args.label,
        'source_h5': args.saved_h5,
        'epoch': args.epoch,
        'mode': args.mode,
        'n_data': int(N),
        'n_steps': int(T),
        'hidden_dim': int(D),
        'n_samples': int(N * T),
        'h1_to_self': r2_h1_self,
        'h1_to_other': r2_h1_other,
        'h2_to_self': r2_h2_self,
        'h2_to_other': r2_h2_other,
    }
    result.update(util.gen_result_metadata(
        exp_config_name=exp_config_name, seed=seed, dataset_name=dataset_name))

    txt = (
        f'=== {args.label} R^2 (epoch {args.epoch}, mode {args.mode}) ===\n'
        f'source: {args.saved_h5}\n'
        f'n_samples: {N * T} (N={N} x T={T})\n'
        f'h1 -> self  pos: {r2_h1_self:.4f}\n'
        f'h1 -> other pos: {r2_h1_other:.4f}\n'
        f'h2 -> self  pos: {r2_h2_self:.4f}\n'
        f'h2 -> other pos: {r2_h2_other:.4f}\n'
    )
    print(txt)

    with open(os.path.join(SAVE_DIR, f'{args.label}_r2.txt'), 'w') as f:
        f.write(txt)
    with open(os.path.join(SAVE_DIR, f'{args.label}_r2.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f'Saved: {SAVE_DIR}/{args.label}_r2.{{txt,json}}')


if __name__ == '__main__':
    main()
