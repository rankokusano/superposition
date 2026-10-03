"""
S5 follow-up (2026-10-04, user request): bar chart of r_hat's 4 components
(mean +/- sd) against A-1's and A-2's true reward vectors, to make visible
what the single cos_sim number hides -- docs/v6_experiment_log.md Sec.39.2
originally reported "closer to A2" from cos_sim alone, but the full
component breakdown shows blue/cyan (which should be ~0 in EITHER true
vector) are both large and positive, and red (which should be negative to
match A-2) is actually weakly POSITIVE, not negative. The A2-ward cos_sim
comes only from green being somewhat higher than red, riding on top of a
large shared positive bias across all 4 components -- not a clean
"identified A-2's preference" signal.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/plot_r_hat_components_v6.py \
        --saved_h5 data/result/v6_s5_ve/0/test/r_hat_trend/save/saved.h5 \
        --epoch 400 --label v6_s5_ve
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = 'data/result/v6_baseline'
LABELS = ['red', 'green', 'blue', 'cyan']
A1_TRUE_R = np.array([1.0, -1.0, 0.0, 0.0])
A2_TRUE_R = np.array([-1.0, 1.0, 0.0, 0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saved_h5', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--mode', default='eval')
    ap.add_argument('--label', required=True)
    args = ap.parse_args()

    with h5py.File(args.saved_h5, 'r') as f:
        r = f[f'{args.epoch:05d}/{args.mode}/r_hat/prediction'][:].reshape(-1, 4)

    mean, sd = r.mean(axis=0), r.std(axis=0)

    x = np.arange(4)
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, A1_TRUE_R, w, label='A-1 true r', color='tab:red', alpha=0.7)
    ax.bar(x, A2_TRUE_R, w, label='A-2 true r', color='tab:green', alpha=0.7)
    ax.bar(x + w, mean, w, yerr=sd, capsize=4, label='r_hat (mean +/- sd)', color='tab:orange')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel('reward weight')
    ax.set_ylim(-1.3, 1.3)
    ax.legend()
    ax.set_title(f'{args.label} ep{args.epoch}: r_hat components vs true reward vectors (n={len(r)})')
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{args.label}_r_hat_components_ep{args.epoch}.png')
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f'Saved: {out_path}')
    for l, m, s in zip(LABELS, mean, sd):
        print(f'  {l:5s}: mean={m:+.4f}  sd={s:.4f}')


if __name__ == '__main__':
    main()
