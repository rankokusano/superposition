"""
Figures for v7 stage 2 (docs/v7_experiment_log.md §2.8, §2.9):
  1. accuracy vs t: stage 1, stage 2 (a)/(b), state controls z1-z3, and the
     colour / self-projection baselines for (a) and (b)
  2. margin-sign agreement (true vision vs v̂²) per beta, near/far, (a) and (b)
  3. distribution of the critic's best direction per r: true vision, v̂²(a), v̂²(b)

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/plot_irl_stage2_v7.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/work/my_research/preference_inference/analyze')
import irl_common_v7 as C  # noqa

R = os.path.join(C.PI_ROOT, 'data', 'result', 'v7_irl')
D = os.path.join(R, 'stage2')


def acc(res, key):
    return [res[key][str(t)]['acc'] for t in C.T_EVAL]


def main():
    s1 = json.load(open(os.path.join(R, 'stage1', 'stage1_results.json')))
    sa = json.load(open(os.path.join(D, 'stage2_a_results.json')))
    sb = json.load(open(os.path.join(D, 'stage2_b_results.json')))
    z = json.load(open(os.path.join(D, 'stage2_state_controls.json')))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for res, key, lab, c, ls in [
            (s1, 'irl_main', 'stage 1: true A-2 vision', 'k', '-'),
            (sa, 'irl_main', 'stage 2 (a): v̂², paper Encoder', 'tab:red', '-'),
            (sb, 'irl_main', 'stage 2 (b): v̂², v6 Encoder', 'tab:blue', '-'),
            (z, 'z1_gray', 'control z1: uniform gray state', 'tab:gray', '--'),
            (z, 'z2_shuffled_true_vision', 'control z2: other episode\'s vision', 'tab:olive', '--'),
            (z, 'z3_geometric_direction', 'control z3: no critic, fixed direction', 'tab:purple', ':')]:
        ax[0].plot(C.T_EVAL, acc(res, key), marker='o', color=c, ls=ls, label=lab)
    ax[0].axhline(1 / 3, color='gray', lw=0.8, label='chance = always r¹')
    ax[0].set(xscale='log', xlabel='t', ylabel='test accuracy (N=300)', ylim=(0, 1.05),
              title='IRL with true action: does the state matter?')
    ax[0].legend(fontsize=8, loc='lower right')
    for res, lab, c in [(sa, '(a)', 'tab:red'), (sb, '(b)', 'tab:blue')]:
        ax[1].plot(C.T_EVAL, acc(res, 'colour_cumulative'), marker='s', color=c, ls='-', label=f'{lab} colour cumulative on v̂²')
        ax[1].plot(C.T_EVAL, acc(res, 'self_projection_prior'), marker='^', color=c, ls='--', label=f'{lab} IRL, prior P(r¹)=0.5')
        ax[1].plot(C.T_EVAL, acc(res, 'v_slope'), marker='x', color=c, ls=':', label=f'{lab} action-free V slope')
    ax[1].plot(C.T_EVAL, acc(sa, 'position_baseline'), marker='o', color='k', ls=':', label='position baseline (true pos.)')
    ax[1].axhline(1 / 3, color='gray', lw=0.8)
    ax[1].set(xscale='log', xlabel='t', ylim=(0, 1.05), title='Stage 2 baselines')
    ax[1].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(D, 'stage2_accuracy_vs_t.png'), dpi=120)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    betas = [str(b) for b in C.BETA_MAIN]
    for i, (res, lab) in enumerate([(sa, '(a) paper Encoder'), (sb, '(b) v6 Encoder')]):
        m = res['q_agreement']['margin_sign_agreement']
        ax[i].plot(betas, [m[b]['all'] for b in betas], 'k-o', label='all frames')
        ax[i].plot(betas, [m[b]['far_ge5'] for b in betas], 'b--o', label='≥5 from goal')
        ax[i].plot(betas, [m[b]['near_lt5'] for b in betas], 'r--o', label='<5 from goal')
        for g, c in zip(C.CAND_NAMES[:3], ('#d62728', '#2ca02c', '#17becf')):
            ax[i].plot(betas, [m[b]['by_goal'][g]['all'] for b in betas], color=c, lw=0.8, label=f'goal {g}')
        ax[i].set(xlabel='β', title=f'margin-sign agreement, true vision vs v̂² {lab}', ylim=(0, 1.02))
        ax[i].legend(fontsize=7)
    ax[0].set_ylabel('fraction of frames with the same sign')
    fig.tight_layout()
    fig.savefig(os.path.join(D, 'stage2_margin_sign_agreement.png'), dpi=120)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=True)
    deg = np.arange(C.K_DIRS) * 360 / C.K_DIRS
    for i, (path, lab) in enumerate([(os.path.join(R, 'stage1', 'q_stage1.npz'), 'true A-2 vision'),
                                     (os.path.join(D, 'q_stage2_a.npz'), 'v̂² (a)'),
                                     (os.path.join(D, 'q_stage2_b.npz'), 'v̂² (b)')]):
        q = np.load(path)['q64']
        for j, c in zip(range(3), ('#d62728', '#2ca02c', '#17becf')):
            k = q[:, :, j].argmax(-1).ravel()
            ax[i].bar(deg, np.bincount(k, minlength=C.K_DIRS) / k.size, width=5, color=c, alpha=0.6,
                      label=f'r = {C.CAND_NAMES[j]}')
        for th, c in zip((135, 225, 45), ('#d62728', '#2ca02c', '#17becf')):
            ax[i].axvline(th, color=c, ls=':', lw=1)
        ax[i].set(title=f'critic best direction, state = {lab}', xlabel='direction (deg, 0=east, 90=north)',
                  xticks=range(0, 361, 45))
    ax[0].set_ylabel('fraction of test frames')
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(D, 'stage2_best_direction_hist.png'), dpi=120)
    print('Saved figures to', D)


if __name__ == '__main__':
    main()
