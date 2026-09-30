"""
S4 required figures (docs/v6_instructions.md Sec.10.3 "S4" row): the
4-condition ablation bar chart (self_vision L1 loss) and the 4-axis R^2
bars per condition, from analyze/oracle_eval_v6.py's saved JSON.

h1->self/h1->other are invariant across the 4 ablation modes (process-1 is
untouched by the process-2 substitution), so they are passed in as fixed
reference values (from the S3 canonical-protocol result) rather than
re-derived here.

Usage:
    python analyze/plot_oracle_bars_v6.py \
        --result data/result/v6_baseline/v6_s3_base_l1_oracle.json \
        --label v6_s3_base_l1_oracle \
        --h1_to_self 0.4730 --h1_to_other 0.4718
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = 'data/result/v6_baseline'
MODES = ['true_r', 'wrong_r', 'zero', 'constant']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--h1_to_self', type=float, required=True)
    ap.add_argument('--h1_to_other', type=float, required=True)
    args = ap.parse_args()

    with open(args.result) as f:
        d = json.load(f)

    # --- Fig A: 4-condition ablation bar (self_vision L1), both ov_enc sources ---
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(MODES))
    w = 0.35
    other_vals = [d['vision_l1_by_condition'][f'other_vision__{m}'] for m in MODES]
    self_vals = [d['vision_l1_by_condition'][f'self_vision__{m}'] for m in MODES]
    ax.bar(x - w / 2, other_vals, w, label='ov_enc=other_vision (primary)')
    ax.bar(x + w / 2, self_vals, w, label='ov_enc=self_vision (secondary)')
    ax.set_xticks(x)
    ax.set_xticklabels(MODES)
    ax.set_ylabel('self_vision L1 loss (lower=better)')
    ax.set_title(f'{args.label}: S4 4-condition ablation')
    ax.legend()
    fig.tight_layout()
    out1 = os.path.join(SAVE_DIR, f'{args.label}_ablation_bars.png')
    fig.savefig(out1, dpi=130)
    plt.close(fig)
    print('saved', out1)

    # --- Fig B: 4-axis R^2 per mode (primary ov_enc=other_vision) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    axes = ['h1_to_self', 'h1_to_other', 'h2_to_self', 'h2_to_other']
    xw = np.arange(len(axes))
    bw = 0.8 / len(MODES)
    for i, mode in enumerate(MODES):
        h2 = d['h2_r2_by_mode'][mode]
        vals = [args.h1_to_self, args.h1_to_other, h2['h2_to_self'], h2['h2_to_other']]
        ax.bar(xw + (i - 1.5) * bw, vals, bw, label=mode)
    ax.set_xticks(xw)
    ax.set_xticklabels(axes)
    ax.set_ylabel('R^2')
    ax.set_title(f'{args.label}: 4-axis R^2 by oracle condition (ov_enc=other_vision)')
    ax.legend()
    fig.tight_layout()
    out2 = os.path.join(SAVE_DIR, f'{args.label}_r2_by_condition.png')
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    print('saved', out2)

    # --- Fig C: distance-binned true_r - zero delta (SS28.2 follow-up) ---
    bins = d['distance_binned_ablation']
    labels = ['<5', '5-10', '10-15', '>=15']
    deltas = [bins[l]['true_r_minus_zero'] for l in labels]
    ns = [bins[l]['n'] for l in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['tab:red' if v > 0 else 'tab:green' for v in deltas]
    ax.bar(labels, deltas, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    for i, (v, n) in enumerate(zip(deltas, ns)):
        ax.text(i, v, f'n={n}', ha='center', va='bottom' if v > 0 else 'top', fontsize=8)
    ax.set_ylabel('true_r - zero  (self_vision L1, negative=true_r helps)')
    ax.set_xlabel('distance to Green')
    ax.set_title(f'{args.label}: SS28.2 distance-binned true_r effect')
    fig.tight_layout()
    out3 = os.path.join(SAVE_DIR, f'{args.label}_distance_binned.png')
    fig.savefig(out3, dpi=130)
    plt.close(fig)
    print('saved', out3)


if __name__ == '__main__':
    main()
