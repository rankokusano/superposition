"""
Aggregate v10 stage A per-checkpoint JSONs (stageA_regression.json) into
late-5 mean +- inter-checkpoint sd per seed, plus the mean over seeds; draw
the 4-axis R^2 bar figure (regressors (i) and (ii)) and the self-vision
error bars.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, git_hash  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

AX = ['h1_to_self', 'h1_to_other', 'h2_to_self', 'h2_to_other']


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f'{prefix}{k}/'))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f'{prefix}{k}'] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    args = ap.parse_args()
    base = os.path.join(OUT_ROOT, 'stageA', 'eval', args.exp_config)

    per_seed = {}
    for s in args.seeds:
        rows = []
        for e in args.epochs:
            p = os.path.join(base, f's{s}', f'ep{e}', 'stageA_regression.json')
            r = json.load(open(p))
            r.pop('meta')
            rows.append(flatten(r))
        keys = rows[0].keys()
        per_seed[s] = {k: {'mean': float(np.mean([r[k] for r in rows])),
                           'sd_inter_ckpt': float(np.std([r[k] for r in rows], ddof=1)),
                           'values': [r[k] for r in rows]} for k in keys}
    keys = per_seed[args.seeds[0]].keys()
    over = {k: {'mean_over_seeds': float(np.mean([per_seed[s][k]['mean'] for s in args.seeds])),
                'sd_between_seeds': float(np.std([per_seed[s][k]['mean'] for s in args.seeds], ddof=1))
                if len(args.seeds) > 1 else None} for k in keys}
    out = {'exp_config': args.exp_config, 'seeds': args.seeds, 'epochs': args.epochs,
           'git_commit': git_hash(), 'per_seed_late5': {str(s): v for s, v in per_seed.items()},
           'over_seeds': over}
    with open(os.path.join(base, 'stageA_late5_summary.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # 4-axis R^2 bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True, sharey=True)
    w = 0.38
    x = np.arange(len(AX))
    panels = [('(i) refit on each condition (train split, in-sample)',
               [('A0 original', 'refit_in_sample/A0/', '#4C78A8'), ('A1 swap', 'refit_in_sample/A1/', '#F58518')]),
              ('(ii) regressor fitted on A0, applied unchanged',
               [('A0 original (in-sample)', 'refit_in_sample/A0/', '#4C78A8'),
                ('A1 swap', 'A0_regressor_on_A1_train/', '#F58518')])]
    for ax, (title, series) in zip(axes, panels):
        for i, (lab, pre, col) in enumerate(series):
            means = [over[pre + a]['mean_over_seeds'] for a in AX]
            seeds_vals = [[per_seed[s][pre + a]['mean'] for s in args.seeds] for a in AX]
            sds = [per_seed[args.seeds[0]][pre + a]['sd_inter_ckpt'] for a in AX]
            ax.bar(x + (i - 0.5) * w, means, w * 0.92, color=col, label=lab)
            for j, sv in enumerate(seeds_vals):
                ax.scatter([x[j] + (i - 0.5) * w] * len(sv), sv, s=14, c='k', zorder=3)
            ax.errorbar(x + (i - 0.5) * w, means, yerr=sds, fmt='none', ecolor='k', capsize=3, lw=1)
            for j, m in enumerate(means):
                ax.text(x[j] + (i - 0.5) * w, max(m, 0) + 0.02, f'{m:.3f}', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(['h1->self', 'h1->other', 'h2->self', 'h2->other'])
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.legend(fontsize=8, loc='upper center')
        ax.grid(axis='y', color='0.9')
    axes[0].set_ylabel('R^2 (late-5 mean over seeds; dots = seeds; bar = inter-ckpt sd)')
    fig.suptitle(f'v10 stage A: {args.exp_config} seeds {args.seeds}, ep {args.epochs[0]}-{args.epochs[-1]}')
    fig.savefig(os.path.join(base, 'stageA_r2_bars.png'), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    for i, (cond, col) in enumerate((('A0', '#4C78A8'), ('A1', '#F58518'))):
        for j, mode in enumerate(('eval', 'test')):
            k = f'self_vision_loss/{cond}/{mode}'
            m = over[k]['mean_over_seeds']
            ax.bar(j + (i - 0.5) * 0.38, m, 0.35, color=col, label=f'{cond}' if j == 0 else None)
            ax.scatter([j + (i - 0.5) * 0.38] * len(args.seeds), [per_seed[s][k]['mean'] for s in args.seeds],
                       s=14, c='k', zorder=3)
            ax.text(j + (i - 0.5) * 0.38, m * 1.01, f'{m:.2f}', ha='center', fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['train split (eval group)', 'test split'])
    ax.set_ylabel('self-vision L1 loss (eval loop, summed over t)')
    ax.set_title(f'{args.exp_config}: process-1 self-vision prediction error', fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(base, 'stageA_vision_error_bars.png'), dpi=110)
    plt.close(fig)
    print(json.dumps(over, indent=1))


if __name__ == '__main__':
    main()
