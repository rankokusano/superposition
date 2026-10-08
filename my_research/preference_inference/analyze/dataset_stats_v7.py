"""
v7 stage-0 dataset statistics (docs/v7_experiment_log.md §2.1, §2.5, §2.6).

Per goal and split: episode count, A-2's initial distance to its goal
(distribution, and the stage-1 bins <5 / 5-10 / 10-15 / >=15), terminal and
minimum distance, fraction of frames within 5 of the goal, and the fraction
of frames whose A-2 action magnitude is < 0.1 (excluded from the likelihood
in stages 1-2). Also applies the stop rule of §2.6: Cyan's mean terminal
distance in the collected data must be <= 5.

Uses the true A-2 positions and goal labels -- evaluation only.

Usage (inside Docker):
    cd /work/my_research/preference_inference
    python analyze/dataset_stats_v7.py --dataset v7_a1random_a2goal3
"""
import argparse
import json
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PI_ROOT = '/work/my_research/preference_inference'
SAVE_DIR = os.path.join(PI_ROOT, 'data', 'result', 'v7_irl', 'stage0')
GATE_DIST = 5.0
SMALL_ACTION = 0.1
DIST_BINS = [(0, 5), (5, 10), (10, 15), (15, np.inf)]
BIN_NAMES = ['<5', '5-10', '10-15', '>=15']


def stats_for(pos, motion, goal_pos):
    d = np.linalg.norm(pos - goal_pos[None, None], axis=-1)  # (n, T)
    init = d[:, 0]
    a = np.linalg.norm(motion, axis=-1)
    return dict(
        n_episodes=int(len(d)),
        init_mean=float(init.mean()), init_sd=float(init.std()),
        init_quantiles={q: float(np.quantile(init, q / 100)) for q in (0, 10, 25, 50, 75, 90, 100)},
        init_bin_counts={name: int(((init >= lo) & (init < hi)).sum())
                         for name, (lo, hi) in zip(BIN_NAMES, DIST_BINS)},
        final_mean=float(d[:, -1].mean()), final_sd=float(d[:, -1].std()),
        min_mean=float(d.min(1).mean()), min_sd=float(d.min(1).std()),
        frac_episodes_final_within_5=float((d[:, -1] <= GATE_DIST).mean()),
        frac_frames_within_5=float((d <= GATE_DIST).mean()),
        frac_frames_small_action=float((a < SMALL_ACTION).mean()),
        action_norm_mean=float(a.mean()),
    ), d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    path = os.path.join(PI_ROOT, 'data', 'data', args.dataset, 'data.h5')
    f = h5py.File(path, 'r')
    names = f.attrs['goal_names'].split(',')
    goal_pos = np.asarray(f.attrs['goal_pos'])

    out = {'dataset': args.dataset, 'path': path, 'seed': int(f.attrs['seed']),
           'goal_names': names, 'splits': {}}
    curves = {}
    for split in ('train', 'test'):
        g = f[f'{split}/goal_index'][()]
        pos = f[f'{split}/other_position'][()]
        motion = f[f'{split}/other_motion'][()]
        out['splits'][split] = {}
        for k, name in enumerate(names):
            s, d = stats_for(pos[g == k], motion[g == k], goal_pos[k])
            out['splits'][split][name] = s
            curves[(split, name)] = d
            print(f'[{split}] {name:5s} n={s["n_episodes"]:4d}  init={s["init_mean"]:.2f}+-{s["init_sd"]:.2f} '
                  f'bins={s["init_bin_counts"]}  final={s["final_mean"]:.2f}+-{s["final_sd"]:.2f}  '
                  f'min={s["min_mean"]:.2f}+-{s["min_sd"]:.2f}  ep_final<=5={s["frac_episodes_final_within_5"]:.3f}  '
                  f'frames<=5={s["frac_frames_within_5"]:.3f}  |a|<0.1={s["frac_frames_small_action"]:.4f}')

    # Stop rule (§2.6): Cyan must reach its goal in the collected data.
    stop = {split: out['splits'][split]['Cyan']['final_mean'] <= GATE_DIST for split in ('train', 'test')}
    out['cyan_reaches_goal'] = stop
    print(f'Cyan reaches goal in collected data (mean final <= {GATE_DIST}): {stop}')

    os.makedirs(SAVE_DIR, exist_ok=True)
    json_path = os.path.join(SAVE_DIR, f'dataset_stats_{args.dataset}.json')
    with open(json_path, 'w') as fp:
        json.dump(out, fp, indent=2)
    print(f'Saved: {json_path}')

    colors = {'Red': '#d62728', 'Green': '#2ca02c', 'Cyan': '#17becf'}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name in names:
        d = curves[('train', name)]
        ax[0].hist(d[:, 0], bins=np.arange(0, 29, 1), histtype='step', color=colors[name], label=name)
        ax[1].plot(d.mean(0), color=colors[name], label=name)
        ax[1].fill_between(np.arange(d.shape[1]), np.quantile(d, 0.25, 0), np.quantile(d, 0.75, 0),
                           color=colors[name], alpha=0.15)
        ax[2].plot((d <= GATE_DIST).mean(0), color=colors[name], label=name)
    ax[0].set(xlabel='initial distance to goal', ylabel='episodes', title='train: initial distance')
    ax[1].set(xlabel='frame', ylabel='distance to goal', title='train: distance (mean, IQR)')
    ax[2].set(xlabel='frame', ylabel='fraction of episodes', title='train: within 5 of goal', ylim=(0, 1))
    for a in ax:
        a.legend()
    fig.tight_layout()
    png_path = os.path.join(SAVE_DIR, f'dataset_stats_{args.dataset}.png')
    fig.savefig(png_path, dpi=120)
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    main()
