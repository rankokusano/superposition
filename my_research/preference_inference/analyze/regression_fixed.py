"""
Joint 4-class regression analysis on FixedA2 test datasets.

Loads saved.h5 from each of 4 fixed-landmark datasets, concatenates hidden
states + labels, then runs logistic regression and produces PCA scatter plot.

If accuracy is high (≥95%): model encodes preference, not cycle order.
If accuracy is low (<70%): model learned cyclic pattern, not static preference.

Usage (from preference_inference/ dir):
    python analyze/regression_fixed.py \
        --result_base data/result/new_exp_b_mg_ve/0/test \
        --epoch 200 --mode eval --layer other
"""

import argparse
import os
import sys

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LANDMARK_NAMES = ['Red', 'Green', 'Blue', 'Yellow']
COLORS = ['red', 'green', 'blue', 'gold']


def load_hidden_and_labels(saved_h5_path, epoch, mode, layer):
    """Load (N*T, D) hidden states and (N*T,) labels from saved.h5."""
    with h5py.File(saved_h5_path, 'r') as f:
        h_key = f'{epoch:05d}/{mode}/state/{layer}/hidden'
        l_key = f'{epoch:05d}/{mode}/a2_target_landmark/input'
        h = f[h_key][()]   # (N, T, D)
        y = f[l_key][()]   # (N, T)
    N, T, D = h.shape
    return h.reshape(N * T, D), y.reshape(N * T).astype(int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_base', required=True,
                        help='e.g. data/result/new_exp_b_mg_ve/0/test')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--layer', default='other')
    args = parser.parse_args()

    all_h, all_y = [], []

    for i, lm in enumerate(LANDMARK_NAMES):
        data_name = f'test_self_rl_other_fixed_{lm.lower()}'
        saved_h5 = os.path.join(
            args.result_base, data_name, 'save', 'saved.h5')
        if not os.path.exists(saved_h5):
            print(f'ERROR: missing {saved_h5}', file=sys.stderr)
            sys.exit(1)
        h, y = load_hidden_and_labels(saved_h5, args.epoch, args.mode, args.layer)
        all_h.append(h)
        all_y.append(y)
        print(f'  {lm}: {len(h)} samples, labels={np.unique(y).tolist()}')

    H = np.concatenate(all_h, axis=0)   # (4*N*T, D)
    Y = np.concatenate(all_y, axis=0)   # (4*N*T,)

    # 4-class logistic regression
    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
    clf.fit(H, Y)
    acc = accuracy_score(Y, clf.predict(H))

    # Ridge regression (continuous proxy for class separation)
    reg = Ridge(alpha=1.0)
    reg.fit(H, Y.astype(float))
    r2 = r2_score(Y.astype(float), reg.predict(H))

    result_str = (
        f'layer: {args.layer}\n'
        f'mode: {args.mode}\n'
        f'epoch: {args.epoch}\n'
        f'n_samples: {len(H)}\n'
        f'logistic_accuracy: {acc}\n'
        f'ridge_r2: {r2}\n'
    )
    print(result_str)

    # Save text result
    save_dir = os.path.normpath(
        os.path.join(args.result_base, '..', 'regression_fixed', args.mode))
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, f'{args.epoch:05d}_layer-{args.layer}.txt')
    with open(txt_path, 'w') as f:
        f.write(result_str)
    print(f'Saved: {txt_path}')

    # PCA visualization
    pca = PCA(n_components=2)
    H2 = pca.fit_transform(H)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (lm, col) in enumerate(zip(LANDMARK_NAMES, COLORS)):
        mask = Y == i
        ax.scatter(H2[mask, 0], H2[mask, 1],
                   c=col, label=lm, alpha=0.3, s=1, rasterized=True)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(
        f'FixedA2 — {args.layer} layer  epoch={args.epoch}  acc={acc:.3f}')
    ax.legend(markerscale=5)

    plot_dir = os.path.normpath(
        os.path.join(args.result_base, '..', 'plots_fixed', args.mode))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir, f'{args.epoch:05d}_layer-{args.layer}_pca.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {plot_path}')
