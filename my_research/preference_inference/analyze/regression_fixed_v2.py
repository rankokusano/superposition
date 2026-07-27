"""
Joint 4-class regression on v2 FixedA2 test datasets.

Same structure as regression_fixed.py but uses v2_fixed_* dataset names.

Usage (from preference_inference/ dir):
    python analyze/regression_fixed_v2.py \\
        --result_base data/result/v2_exp_b_mg_ve/0/test \\
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
    with h5py.File(saved_h5_path, 'r') as f:
        h_key = f'{epoch:05d}/{mode}/state/{layer}/hidden'
        l_key = f'{epoch:05d}/{mode}/a2_target_landmark/input'
        h = f[h_key][()]
        y = f[l_key][()]
    N, T, D = h.shape
    return h.reshape(N * T, D), y.reshape(N * T).astype(int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_base', required=True,
                        help='e.g. data/result/v2_exp_b_mg_ve/0/test')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--layer', default='other')
    args = parser.parse_args()

    all_h, all_y = [], []

    for lm in LANDMARK_NAMES:
        data_name = f'v2_fixed_{lm.lower()}'
        saved_h5 = os.path.join(args.result_base, data_name, 'save', 'saved.h5')
        if not os.path.exists(saved_h5):
            print(f'ERROR: missing {saved_h5}', file=sys.stderr)
            sys.exit(1)
        h, y = load_hidden_and_labels(saved_h5, args.epoch, args.mode, args.layer)
        all_h.append(h)
        all_y.append(y)
        print(f'  {lm}: {len(h)} samples, labels={np.unique(y).tolist()}')

    H = np.concatenate(all_h, axis=0)
    Y = np.concatenate(all_y, axis=0)

    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
    clf.fit(H, Y)
    acc = accuracy_score(Y, clf.predict(H))

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

    save_dir = os.path.normpath(
        os.path.join(args.result_base, '..', 'regression_fixed_v2', args.mode))
    os.makedirs(save_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, f'{args.epoch:05d}_layer-{args.layer}.txt')
    with open(txt_path, 'w') as f:
        f.write(result_str)
    print(f'Saved: {txt_path}')

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
        f'v2 FixedA2 — {args.layer} layer  epoch={args.epoch}  acc={acc:.3f}')
    ax.legend(markerscale=5)

    plot_dir = os.path.normpath(
        os.path.join(args.result_base, '..', 'plots_fixed_v2', args.mode))
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(
        plot_dir, f'{args.epoch:05d}_layer-{args.layer}_pca.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {plot_path}')
