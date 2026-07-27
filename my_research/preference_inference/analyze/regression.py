"""
Regression analysis: decode A-2 target landmark (4-class) from SM hidden state.

Two analyses:
  1. Linear ridge regression on landmark ID (regression) → R²
  2. Logistic regression (multiclass classification) → accuracy

Usage:
    cd /work/my_research/preference_inference/analyze
    python regression.py \
        --result_dir ../data/result/new_exp_a/seed0 \
        --epoch 200 \
        --mode eval \
        --layer other
"""

import argparse
import os

import numpy
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

from data_loader import DataLoader

LANDMARK_NAMES = ['Red', 'Green', 'Blue', 'Yellow']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--state_type', default='hidden')
    parser.add_argument('--layer', type=str, default='other',
                        help='SM hidden state layer to decode from (self/other)')
    args = parser.parse_args()

    saved_h5 = os.path.join(args.result_dir, 'saved.h5')
    loader = DataLoader()
    loader.set_data(saved_h5, args.epoch)

    savedir = os.path.join(args.result_dir, 'regression', args.mode)
    os.makedirs(savedir, exist_ok=True)

    # Flatten hidden states: (N*T, hidden_dim)
    h = loader.get_flatten_hidden(mode=args.mode, layer=args.layer,
                                  state_type=args.state_type)

    # Flatten landmark labels: (N*T,) - stored as float, convert to int
    y = loader.get_flatten_a2_landmark_label(mode=args.mode).astype(int)

    # Logistic regression (4-class classification)
    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
    clf.fit(h, y)
    acc = accuracy_score(y, clf.predict(h))

    # Ridge regression on integer label (continuous proxy)
    reg = Ridge(alpha=1.0)
    reg.fit(h, y.astype(float))
    pred_reg = reg.predict(h)
    r2 = r2_score(y.astype(float), pred_reg)

    result = {
        'layer': args.layer,
        'mode': args.mode,
        'epoch': args.epoch,
        'logistic_accuracy': acc,
        'ridge_r2': r2,
    }

    f_name = os.path.join(savedir,
        f'{args.epoch:05d}_layer-{args.layer}.txt')
    with open(f_name, 'w') as f:
        for k, v in result.items():
            f.write(f'{k}: {v}\n')

    print(f"Layer={args.layer} | Logistic Acc={acc:.4f} | Ridge R²={r2:.4f}")
    print(f"Saved: {f_name}")
