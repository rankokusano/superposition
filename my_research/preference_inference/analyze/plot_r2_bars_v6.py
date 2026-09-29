"""
S2 4-axis R^2 bar chart (docs/v6_instructions.md SS10.1/10.3): bars for
h1->self, h1->other, h2->self, h2->other, comparing the in-distribution
(r3_stay) and canonical-protocol (r2_a1random_a2rl) late-5 aggregates.

Usage (no GPU needed):
    python analyze/plot_r2_bars_v6.py \
        --canonical data/result/v6_baseline/v6_s2_base_mse_s0_late5_aggregate_r2_a1random_a2rl.json \
        --label v6_s2_base_mse \
        --indist_h1_self 0.8215 --indist_h1_self_sd 0.00397 \
        --indist_h1_other 0.5400 --indist_h1_other_sd 0.00151 \
        --indist_h2_self 0.2061 --indist_h2_self_sd 0.00586 \
        --indist_h2_other 0.5176 --indist_h2_other_sd 0.00135
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = 'data/result/v6_baseline'
AXES = ['h1_to_self', 'h1_to_other', 'h2_to_self', 'h2_to_other']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', required=True, help='late5 aggregate json (r2_a1random_a2rl)')
    ap.add_argument('--label', required=True)
    for ax in AXES:
        ap.add_argument(f'--indist_{ax}', type=float, required=True)
        ap.add_argument(f'--indist_{ax}_sd', type=float, required=True)
    args = ap.parse_args()

    with open(args.canonical) as f:
        canon = json.load(f)['r2']

    canon_mean = [float(np.mean(canon[ax])) for ax in AXES]
    canon_sd = [float(np.std(canon[ax])) for ax in AXES]
    indist_mean = [getattr(args, f'indist_{ax}') for ax in AXES]
    indist_sd = [getattr(args, f'indist_{ax}_sd') for ax in AXES]

    x = np.arange(len(AXES))
    w = 0.35
    fig, ax_ = plt.subplots(figsize=(7, 5))
    ax_.bar(x - w / 2, indist_mean, w, yerr=indist_sd, capsize=4, label='r3_stay (in-distribution)')
    ax_.bar(x + w / 2, canon_mean, w, yerr=canon_sd, capsize=4, label='r2_a1random_a2rl (canonical)')
    ax_.set_xticks(x)
    ax_.set_xticklabels(AXES)
    ax_.set_ylabel('R^2 (late-5 mean +/- sd)')
    ax_.set_title(f'{args.label}: 4-axis R^2')
    ax_.legend()
    ax_.set_ylim(0, 1)
    fig.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    out = os.path.join(SAVE_DIR, f'{args.label}_r2_bars.png')
    fig.savefig(out, dpi=130)
    print('saved', out)


if __name__ == '__main__':
    main()
