"""
Plot training/eval loss curves for all 3 v3 stages.

Usage (from /work/my_research/preference_inference inside Docker):
    python analyze/plot_loss_v3.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULT_ROOT = 'data/result'
SAVE_DIR = 'data/result/v3_loss_curves'

STAGES = [
    ('Stage 1: B-base MSE', 'v3_exp_b_base_mse'),
    ('Stage 2: B-base L1',  'v3_exp_b_base_l1'),
    ('Stage 3: B-MGVE',     'v3_exp_b_mgve'),
]

LOSS_KEYS = ['self_vision', 'feature_prediction_self', 'feature_prediction_other']
LOSS_LABELS = {
    'self_vision': 'Vision reconstruction',
    'feature_prediction_self': 'FPM self',
    'feature_prediction_other': 'FPM other',
}
COLORS = {'self_vision': 'steelblue', 'feature_prediction_self': 'darkorange', 'feature_prediction_other': 'green'}


def load_log(path):
    if not os.path.exists(path):
        return None
    epochs, vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e, v = line.split(',')
            epochs.append(int(e))
            vals.append(float(v))
    return np.array(epochs), np.array(vals)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig, axes = plt.subplots(len(STAGES), 3, figsize=(15, 4 * len(STAGES)))
    fig.suptitle('v3 Training Loss Curves', fontsize=14)

    for row, (stage_label, exp_name) in enumerate(STAGES):
        log_base = os.path.join(RESULT_ROOT, exp_name, '0', 'log')

        for col, key in enumerate(LOSS_KEYS):
            ax = axes[row, col]
            ax.set_title(f'{stage_label}\n{LOSS_LABELS[key]}', fontsize=9)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

            for split, ls in [('train', '-'), ('eval', '--')]:
                log_path = os.path.join(log_base, split, f'{key}.log')
                result = load_log(log_path)
                if result is None:
                    continue
                epochs, vals = result
                ax.plot(epochs, vals, ls, color=COLORS[key],
                        label=split, alpha=0.85, linewidth=1.2)

            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'loss_curves_v3.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')

    # Print final loss values
    print('\n=== Final loss values (epoch 200) ===')
    for stage_label, exp_name in STAGES:
        print(f'\n{stage_label}:')
        log_base = os.path.join(RESULT_ROOT, exp_name, '0', 'log')
        for key in LOSS_KEYS:
            for split in ['train', 'eval']:
                log_path = os.path.join(log_base, split, f'{key}.log')
                result = load_log(log_path)
                if result is None:
                    continue
                print(f'  {split:5s} {key}: {result[1][-1]:.4f}')


if __name__ == '__main__':
    main()
