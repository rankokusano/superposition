"""
v4 R2 diagnosis (2026-08-14): does A-2's motion vector have enough
directional diversity for MG to learn from, compared to A-1 (RandomAgent,
reference for "MG-learnable" conditions) and to exp3's Cycler A-2?

Hypothesis under test: A-2's RL policy in R2 moves almost monotonically
toward Green (mostly negative y), unlike exp3's Cycler which tours all 4
landmarks and so covers all directions -- MG may simply lack the training
signal diversity to learn a real mapping, unlike position coverage (which
was the earlier, now-separate concern).

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/compare_motion_variance.py
"""
import json
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = 'data/result/baseline_v4'

SOURCES = {
    'r2_a2_rl': '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5',
    'exp3': '/work/data/data/self_random_other_stay_periodic/data.h5',
}


def load_motion(h5_path, mode='train'):
    with h5py.File(h5_path, 'r') as f:
        sm = f[f'{mode}/self_motion'][()].reshape(-1, 2)
        om = f[f'{mode}/other_motion'][()].reshape(-1, 2)
    return sm, om


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    results = {}

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for col, (label, path) in enumerate(SOURCES.items()):
        sm, om = load_motion(path)

        results[label] = {
            'self_motion': {
                'mean': sm.mean(axis=0).tolist(), 'std': sm.std(axis=0).tolist(),
                'min': sm.min(axis=0).tolist(), 'max': sm.max(axis=0).tolist(),
            },
            'other_motion': {
                'mean': om.mean(axis=0).tolist(), 'std': om.std(axis=0).tolist(),
                'min': om.min(axis=0).tolist(), 'max': om.max(axis=0).tolist(),
            },
        }

        print(f'\n=== {label} ===')
        print(f'  self_motion  (A-1): mean={sm.mean(0)}  std={sm.std(0)}  '
              f'range_x=[{sm[:,0].min():.3f},{sm[:,0].max():.3f}]  range_y=[{sm[:,1].min():.3f},{sm[:,1].max():.3f}]')
        print(f'  other_motion (A-2): mean={om.mean(0)}  std={om.std(0)}  '
              f'range_x=[{om[:,0].min():.3f},{om[:,0].max():.3f}]  range_y=[{om[:,1].min():.3f},{om[:,1].max():.3f}]')

        axes[0, col].hist(sm[:, 0], bins=60, alpha=0.6, label='x', color='steelblue')
        axes[0, col].hist(sm[:, 1], bins=60, alpha=0.6, label='y', color='darkorange')
        axes[0, col].set_title(f'{label}: A-1 self_motion (std x={sm[:,0].std():.3f}, y={sm[:,1].std():.3f})')
        axes[0, col].legend()

        axes[1, col].hist(om[:, 0], bins=60, alpha=0.6, label='x', color='steelblue')
        axes[1, col].hist(om[:, 1], bins=60, alpha=0.6, label='y', color='darkorange')
        axes[1, col].set_title(f'{label}: A-2 other_motion (std x={om[:,0].std():.3f}, y={om[:,1].std():.3f})')
        axes[1, col].legend()

    # remaining columns unused if only 2 sources; hide them
    for col in range(len(SOURCES), 4):
        axes[0, col].axis('off')
        axes[1, col].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, 'r2_vs_exp3_motion_histograms.png')
    plt.savefig(plot_path, dpi=110)
    plt.close()
    print(f'\nSaved: {plot_path}')

    out_path = os.path.join(SAVE_DIR, 'r2_motion_variance_check.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
