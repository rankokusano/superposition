"""
Figures for v7 stage 1 (docs/v7_experiment_log.md §2.2, §2.7), from
stage1_results.json and q_stage1.npz written by irl_stage1_v7.py.

  1. accuracy vs t for all methods (+ chance), and mean posterior of the
     correct r per t (main IRL)
  2. per-frame margin (correct r log-lik minus best other) near (<5) vs
     far (>=5) from the goal, for each beta of the main grid
  3. Delta Q (max - min over 64 directions, true r) per goal, near vs far

Usage (inside Docker):
    cd /work/my_research/preference_inference
    python analyze/plot_irl_stage1_v7.py
"""
import json
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/work/my_research/preference_inference/analyze')
import irl_common_v7 as C  # noqa

DIR = os.path.join(C.PI_ROOT, 'data', 'result', 'v7_irl', 'stage1')
DATA_PATH = os.path.join(C.PI_ROOT, 'data', 'data', 'v7_a1random_a2goal3', 'data.h5')
COLORS = {'Red': '#d62728', 'Green': '#2ca02c', 'Cyan': '#17becf'}


def main():
    r = json.load(open(os.path.join(DIR, 'stage1_results.json')))
    q = np.load(os.path.join(DIR, 'q_stage1.npz'))
    q64, qobs = q['q64'][:, :, :C.N_MAIN], q['qobs'][:, :, :C.N_MAIN]
    f = h5py.File(DATA_PATH, 'r')
    goal = f['test/goal_index'][()]
    opos = f['test/other_position'][()]
    N, T = qobs.shape[:2]
    near = np.linalg.norm(opos - C.GOAL_POS[goal][:, None], axis=-1) < C.NEAR_DIST

    # 1. accuracy vs t
    ts = [str(t) for t in C.T_EVAL]
    methods = [('irl_main', 'IRL (main, β 0.5–16)', 'k', '-'),
               ('irl_beta_ext', 'IRL (β up to 128)', 'gray', '--'),
               ('self_projection_prior', 'IRL, prior P(r¹)=0.5', 'tab:orange', '-'),
               ('colour_cumulative', 'colour-fraction cumulative', 'tab:purple', '-'),
               ('position_baseline', 'position baseline (true pos.)', 'tab:blue', ':'),
               ('v_slope', 'action-free V slope', 'tab:brown', '-.')]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, label, c, ls in methods:
        ax[0].plot(C.T_EVAL, [r[key][t]['acc'] for t in ts], marker='o', color=c, ls=ls, label=label)
    ax[0].axhline(1 / 3, color='gray', lw=0.8, label='chance 1/3 (= always r¹)')
    ax[0].axhline(0.8, color='red', lw=0.8, ls='--', label='gate 0.8 at t=50')
    ax[0].axvline(50, color='red', lw=0.5, ls=':')
    ax[0].set(xscale='log', xlabel='t (observed state-action pairs)', ylabel='test accuracy',
              ylim=(0, 1.05), title='Stage 1: true A-2 vision + true action (test, N=300)')
    ax[0].legend(fontsize=8, loc='lower right')
    curve = r['mean_post_correct_curve']
    ax[1].plot(np.arange(len(curve)), curve, color='k', label='all goals')
    lp = C.log_posterior_all_t(q64, qobs, np.ones((N, T), bool), C.BETA_MAIN, np.log(np.full(3, 1 / 3)))
    post = np.exp(lp)
    for g, name in enumerate(C.CAND_NAMES[:3]):
        m = goal == g
        ax[1].plot(np.arange(T + 1), post[m, :, g].mean(0), color=COLORS[name], label=name)
    ax[1].axhline(1 / 3, color='gray', lw=0.8)
    ax[1].set(xlabel='t', ylabel='mean posterior of the true r', ylim=(0, 1.05),
              title='Main IRL: posterior of the correct goal')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, 'stage1_accuracy_vs_t.png'), dpi=120)

    # 2. margin near vs far
    fig, ax = plt.subplots(1, len(C.BETA_MAIN), figsize=(3.2 * len(C.BETA_MAIN), 3.4), sharey=False)
    for i, beta in enumerate(C.BETA_MAIN):
        ll = C.frame_loglik(q64, qobs, beta)
        corr = np.take_along_axis(ll, goal[:, None, None].repeat(T, 1), axis=-1)[..., 0]
        oth = ll.copy()
        np.put_along_axis(oth, goal[:, None, None].repeat(T, 1), -np.inf, axis=-1)
        margin = corr - oth.max(-1)
        lo, hi = np.quantile(margin, [0.005, 0.995])
        bins = np.linspace(lo, hi, 60)
        ax[i].hist(margin[~near], bins=bins, density=True, alpha=0.6, label=f'≥5 from goal (n={int((~near).sum())})')
        ax[i].hist(margin[near], bins=bins, density=True, alpha=0.6, label=f'<5 from goal (n={int(near.sum())})')
        ax[i].axvline(0, color='k', lw=0.8)
        ax[i].set(title=f'β={beta}', xlabel='log-lik(correct) − max other')
    ax[0].legend(fontsize=7)
    fig.suptitle('Per-frame margin, near vs far from the goal (stage 1, test)')
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, 'stage1_margin_near_far.png'), dpi=120)

    # 3. Delta Q
    dq = q['q64'].max(-1) - q['q64'].min(-1)
    dq_true = np.take_along_axis(dq, goal[:, None, None].repeat(T, 1), axis=-1)[..., 0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    bins = np.linspace(0, np.quantile(dq_true, 0.999), 60)
    for g, name in enumerate(C.CAND_NAMES[:3]):
        m = goal == g
        ax[g].hist(dq_true[m][~near[m]], bins=bins, density=True, alpha=0.6, label='≥5 from goal')
        ax[g].hist(dq_true[m][near[m]], bins=bins, density=True, alpha=0.6, label='<5 from goal')
        ax[g].set(title=f'goal {name}: ΔQ under true r', xlabel='max_k Q − min_k Q (64 directions)')
        ax[g].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(DIR, 'stage1_delta_q.png'), dpi=120)
    print('Saved figures to', DIR)


if __name__ == '__main__':
    main()
