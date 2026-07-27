"""
h¹/h² → position R² regression for v3 experiments.

Evaluates self-other separation:
  h¹ → A-1 position (self_position)  : should be HIGH
  h² → A-2 position (other_position) : should be HIGH
  h¹ → A-2 position                  : should be LOW
  h² → A-1 position                  : should be LOW

Usage (from /work/my_research/preference_inference inside Docker):
    python analyze/regression_v3.py \
        --result_dir data/result/v3_exp_b_base_l1/0/test/v3_b_mgve_train \
        --epoch 200 --label b_base

    python analyze/regression_v3.py \
        --result_dir data/result/v3_exp_b_mgve/0/test/v3_b_mgve_train \
        --epoch 200 --label b_mgve
"""

import argparse
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

SAVE_DIR = 'data/result/v3_regression'


def load_data(saved_h5, epoch, mode='eval'):
    key_pre = f'{epoch:05d}/{mode}'
    with h5py.File(saved_h5, 'r') as f:
        h1 = f[f'{key_pre}/state/self/hidden'][()]    # (N, T, D)
        h2 = f[f'{key_pre}/state/other/hidden'][()]   # (N, T, D)
        sp = f[f'{key_pre}/self_position/input'][()]  # (N, T, 2)
        op = f[f'{key_pre}/other_position/input'][()]  # (N, T, 2)
    N, T, D = h1.shape
    # flatten time
    h1 = h1.reshape(N * T, D)
    h2 = h2.reshape(N * T, D)
    sp = sp.reshape(N * T, 2)
    op = op.reshape(N * T, 2)
    return h1, h2, sp, op


def r2(X, Y, label_x, label_y):
    reg = Ridge(alpha=1.0)
    reg.fit(X, Y)
    pred = reg.predict(X)
    r2_x = r2_score(Y[:, 0], pred[:, 0])
    r2_y = r2_score(Y[:, 1], pred[:, 1])
    r2_mean = r2_score(Y, pred)
    print(f'  {label_x} → {label_y}: R²_x={r2_x:.4f}, R²_y={r2_y:.4f}, R²_mean={r2_mean:.4f}')
    return r2_mean, reg


def plot_scatter(h, pos, reg, title, save_path):
    pred = reg.predict(h)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for dim, (ax, coord) in enumerate(zip(axes, ['x', 'y'])):
        ax.scatter(pos[:, dim], pred[:, dim], alpha=0.1, s=1, rasterized=True)
        lim = [pos[:, dim].min(), pos[:, dim].max()]
        ax.plot(lim, lim, 'r--', linewidth=1)
        ax.set_xlabel(f'True {coord}')
        ax.set_ylabel(f'Predicted {coord}')
        ax.set_title(f'{coord}-axis')
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', required=True,
                        help='e.g. data/result/v3_exp_b_base_l1/0/test/v3_b_mgve_train')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--label', default='model',
                        help='label for output files (e.g. b_base, b_mgve)')
    args = parser.parse_args()

    saved_h5 = os.path.join(args.result_dir, 'save', 'saved.h5')
    if not os.path.exists(saved_h5):
        raise FileNotFoundError(f'Not found: {saved_h5}')

    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f'\nLoading: {saved_h5}')
    h1, h2, sp, op = load_data(saved_h5, args.epoch)
    print(f'  h1: {h1.shape}, h2: {h2.shape}, sp: {sp.shape}, op: {op.shape}')

    print(f'\n=== R² Regression ({args.label}) ===')
    r2_h1_sp, reg_h1_sp = r2(h1, sp, 'h¹', 'A-1 pos')
    r2_h2_op, reg_h2_op = r2(h2, op, 'h²', 'A-2 pos')
    r2_h1_op, reg_h1_op = r2(h1, op, 'h¹', 'A-2 pos')
    r2_h2_sp, reg_h2_sp = r2(h2, sp, 'h²', 'A-1 pos')

    # Save summary
    summary = (
        f'=== {args.label} R² (epoch {args.epoch}) ===\n'
        f'h¹ → A-1 pos (self):  {r2_h1_sp:.4f}  ← should be HIGH\n'
        f'h² → A-2 pos (other): {r2_h2_op:.4f}  ← should be HIGH\n'
        f'h¹ → A-2 pos:         {r2_h1_op:.4f}  ← should be LOW\n'
        f'h² → A-1 pos:         {r2_h2_sp:.4f}  ← should be LOW\n'
    )
    print(f'\n{summary}')

    txt_path = os.path.join(SAVE_DIR, f'{args.label}_r2.txt')
    with open(txt_path, 'w') as f:
        f.write(summary)
    print(f'Saved: {txt_path}')

    # Scatter plots
    for h, pos, reg, name in [
        (h1, sp, reg_h1_sp, f'{args.label}_h1_to_a1pos'),
        (h2, op, reg_h2_op, f'{args.label}_h2_to_a2pos'),
    ]:
        plot_scatter(h, pos, reg,
                     title=f'{args.label}: {name}',
                     save_path=os.path.join(SAVE_DIR, f'{name}.png'))
        print(f'Saved plot: {name}.png')


if __name__ == '__main__':
    main()
