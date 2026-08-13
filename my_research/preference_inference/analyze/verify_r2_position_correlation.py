"""
v4 R2 result verification (2026-08-13 conversation).

Checks whether R2's h1<->other / h2<->self leakage (h1->other: 0.0129 in
exp3 -> 0.4929 in R2; h2->self: 0.0716 in exp3 -> 0.3421 in R2) is real
self/other confusion in the representation, or just an artifact of A-1
and A-2's true positions being correlated in R2 (both trained to hover
near landmarks sharing the x=-9 wall: Red(-9,9) and Green(-9,-9)) where
exp3's A-2 (Cycler) was positionally independent of A-1.

For both R2 and exp3 data:
    - Pearson correlation of true self_position vs other_position, x and y
      separately
    - Ridge regression baseline: self_position -> other_position R^2, and
      other_position -> self_position R^2 (pure position-to-position, no
      hidden state involved at all -- this is the ceiling any h->other-pos
      or h->self-pos regression could reach for free if the network learned
      nothing about the other agent specifically)
    - position scatter overlay (both agents, same axes)

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/verify_r2_position_correlation.py
"""
import json
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

SAVE_DIR = 'data/result/baseline_v4'

SOURCES = {
    'r2_a2_rl': {
        'h5': '/work/my_research/preference_inference/data/result/r2_a2_rl/0/test/r2_a1random_a2rl/save/saved.h5',
        'epoch': 200,
        'mode': 'eval',
    },
    'exp3': {
        'h5': '/work/data/result/exp3/0/test/self_random_other_stay_periodic/save/saved.h5',
        'epoch': 200,
        'mode': 'eval',
    },
}


def load_positions(h5_path, epoch, mode):
    key_pre = f'{epoch:05d}/{mode}'
    with h5py.File(h5_path, 'r') as f:
        sp = f[f'{key_pre}/self_position/input'][()]
        op = f[f'{key_pre}/other_position/input'][()]
    N, T, _ = sp.shape
    return sp.reshape(N * T, 2), op.reshape(N * T, 2), N, T


def ridge_r2(X, Y):
    reg = Ridge(alpha=1.0)
    reg.fit(X, Y)
    pred = reg.predict(X)
    return float(r2_score(Y, pred))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    results = {}

    fig, axes = plt.subplots(1, len(SOURCES), figsize=(6 * len(SOURCES), 6))
    if len(SOURCES) == 1:
        axes = [axes]

    for (label, cfg), ax in zip(SOURCES.items(), axes):
        sp, op, N, T = load_positions(cfg['h5'], cfg['epoch'], cfg['mode'])

        corr_x = float(np.corrcoef(sp[:, 0], op[:, 0])[0, 1])
        corr_y = float(np.corrcoef(sp[:, 1], op[:, 1])[0, 1])

        r2_self_to_other = ridge_r2(sp, op)  # self_position -> other_position
        r2_other_to_self = ridge_r2(op, sp)  # other_position -> self_position

        print(f'\n=== {label} (N={N}, T={T}, n_samples={N*T}) ===')
        print(f'  corr(self_x, other_x) = {corr_x:.4f}')
        print(f'  corr(self_y, other_y) = {corr_y:.4f}')
        print(f'  Ridge R^2  self_position -> other_position : {r2_self_to_other:.4f}')
        print(f'  Ridge R^2  other_position -> self_position : {r2_other_to_self:.4f}')

        results[label] = {
            'n_data': int(N), 'n_steps': int(T), 'n_samples': int(N * T),
            'corr_self_x_other_x': corr_x,
            'corr_self_y_other_y': corr_y,
            'position_baseline_r2_self_to_other': r2_self_to_other,
            'position_baseline_r2_other_to_self': r2_other_to_self,
        }

        ax.scatter(sp[:, 0], sp[:, 1], s=1, alpha=0.05, color='crimson', label='A-1 (self) true position')
        ax.scatter(op[:, 0], op[:, 1], s=1, alpha=0.05, color='seagreen', label='A-2 (other) true position')
        ax.set_title(f'{label}: true position overlay')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        leg = ax.legend(markerscale=15, loc='upper right')

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, 'r2_vs_exp3_position_overlay.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f'\nSaved plot: {plot_path}')

    print('\n=== Comparison against the regression results already measured ===')
    print('R2:   h1->other R2 = 0.4929 (measured)  vs  position baseline self->other R2 = '
          f'{results["r2_a2_rl"]["position_baseline_r2_self_to_other"]:.4f}')
    print('R2:   h2->self  R2 = 0.3421 (measured)  vs  position baseline other->self R2 = '
          f'{results["r2_a2_rl"]["position_baseline_r2_other_to_self"]:.4f}')
    print('exp3: h1->other R2 = 0.0129 (measured)  vs  position baseline self->other R2 = '
          f'{results["exp3"]["position_baseline_r2_self_to_other"]:.4f}')
    print('exp3: h2->self  R2 = 0.0716 (measured)  vs  position baseline other->self R2 = '
          f'{results["exp3"]["position_baseline_r2_other_to_self"]:.4f}')

    out_path = os.path.join(SAVE_DIR, 'r2_position_correlation_check.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
