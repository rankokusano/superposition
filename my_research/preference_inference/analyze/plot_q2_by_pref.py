"""
Plot VE's estimated Q̂² (q2_hat) over time, grouped by A-2's landmark preference.

For v2 experiments: if A-1 uses own value experience to infer A-2's preference,
then Q̂² should be HIGH when A-2 prefers Red (which A-1 likes)
and LOW when A-2 prefers Green (which A-1 dislikes).

Usage (from preference_inference/ dir):
    python analyze/plot_q2_by_pref.py \\
        --result_base data/result/v2_exp_b_mg_ve/0/test \\
        --epoch 200 \\
        --save_dir data/result/v2_q2_analysis/
"""

import argparse
import os
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LANDMARK_NAMES = ['Red', 'Green', 'Blue', 'Yellow']
COLORS = ['red', 'green', 'blue', 'gold']
PREFIX = 'v2_fixed'


def load_q2_hat(saved_h5_path, epoch, mode='eval'):
    """Load q2_hat from saved.h5. Returns (N, T) array."""
    with h5py.File(saved_h5_path, 'r') as f:
        key = f'{epoch:05d}/{mode}/q2_hat/prediction'
        if key not in f:
            raise KeyError(f'Key not found: {key}')
        q2 = f[key][()]  # (N, T, 1) or (N, T)
    if q2.ndim == 3:
        q2 = q2[..., 0]
    return q2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_base', required=True,
                        help='e.g. data/result/v2_exp_b_mg_ve/0/test')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--mode', default='eval')
    parser.add_argument('--prefix', default='v2_fixed',
                        help='Dataset name prefix (default: v2_fixed)')
    parser.add_argument('--save_dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    per_pref_q2 = {}
    per_pref_mean = {}
    T_common = None

    print(f'Loading q2_hat from: {args.result_base}')
    for lm in LANDMARK_NAMES:
        data_name = f'{args.prefix}_{lm.lower()}'
        saved_h5 = os.path.join(args.result_base, data_name, 'save', 'saved.h5')
        if not os.path.exists(saved_h5):
            print(f'  SKIP (not found): {saved_h5}', file=sys.stderr)
            continue
        q2 = load_q2_hat(saved_h5, args.epoch, args.mode)  # (N, T)
        per_pref_q2[lm] = q2
        per_pref_mean[lm] = q2.mean(axis=0)  # (T,)
        T_common = q2.shape[1]
        print(f'  {lm}: mean Q̂²={q2.mean():.4f}  std={q2.std():.4f}  shape={q2.shape}')

    if not per_pref_mean:
        print('No data found.', file=sys.stderr)
        sys.exit(1)

    # ── Summary bar chart ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    lms_found = [lm for lm in LANDMARK_NAMES if lm in per_pref_mean]
    means   = [per_pref_q2[lm].mean() for lm in lms_found]
    stds    = [per_pref_q2[lm].std()  for lm in lms_found]
    cols    = [COLORS[LANDMARK_NAMES.index(lm)] for lm in lms_found]

    ax.bar(lms_found, means, yerr=stds, color=cols, alpha=0.7, capsize=5)
    ax.set_xlabel('A-2 preference (FixedA2 target)')
    ax.set_ylabel('Mean Q̂² (VE output)')
    ax.set_title(f'Q̂² by A-2 preference  epoch={args.epoch}\n'
                 f'(high Red, low Green → A-1 value experience used for inference)')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    bar_path = os.path.join(args.save_dir, f'{args.epoch:05d}_q2_by_pref_bar.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {bar_path}')

    # ── Time series ──────────────────────────────────────────────────────────
    if T_common is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ts = np.arange(T_common)
        for lm, col in zip(LANDMARK_NAMES, COLORS):
            if lm not in per_pref_mean:
                continue
            mean_t = per_pref_mean[lm]
            std_t  = per_pref_q2[lm].std(axis=0)
            ax.plot(ts, mean_t, color=col, label=lm, linewidth=1.5)
            ax.fill_between(ts, mean_t - std_t, mean_t + std_t,
                            color=col, alpha=0.15)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Q̂² (VE output)')
        ax.set_title(f'Q̂² over time by A-2 preference  epoch={args.epoch}')
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        plt.tight_layout()
        ts_path = os.path.join(args.save_dir, f'{args.epoch:05d}_q2_by_pref_timeseries.png')
        plt.savefig(ts_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {ts_path}')

    # ── Print summary table ───────────────────────────────────────────────────
    print('\n=== Q̂² mean by A-2 preference ===')
    for lm in LANDMARK_NAMES:
        if lm in per_pref_q2:
            print(f'  {lm:8s}: {per_pref_q2[lm].mean():+.4f}')
    if 'Red' in per_pref_q2 and 'Green' in per_pref_q2:
        diff = per_pref_q2['Red'].mean() - per_pref_q2['Green'].mean()
        print(f'\n  Red - Green = {diff:+.4f}')
        if diff > 0:
            print('  ✓ Q̂²(Red) > Q̂²(Green): A-1 uses own value experience to infer A-2 preference')
        else:
            print('  ✗ Q̂²(Red) <= Q̂²(Green): preference inference not confirmed')

    # Save text summary
    summary_path = os.path.join(args.save_dir, f'{args.epoch:05d}_q2_summary.txt')
    with open(summary_path, 'w') as f:
        for lm in LANDMARK_NAMES:
            if lm in per_pref_q2:
                f.write(f'{lm}: {per_pref_q2[lm].mean():+.6f}\n')
        if 'Red' in per_pref_q2 and 'Green' in per_pref_q2:
            diff = per_pref_q2['Red'].mean() - per_pref_q2['Green'].mean()
            f.write(f'Red-Green: {diff:+.6f}\n')
    print(f'Saved: {summary_path}')
