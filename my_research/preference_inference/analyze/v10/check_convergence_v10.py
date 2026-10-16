"""
v10 B3 convergence (docs/v10_experiment_log.md Sec.5.22-5.23, 5.26):
  - v4 criterion on feature_prediction_other (train log): (i) mean(W) <= 1.25 min(B),
    (ii) |slope(W)| |W| <= 0.10 min(B); transition epoch = first ep with fp_other < 1.3 min(B)
  - self-vision criterion on the train-log self_vision / 100 (pixel-wise mean
    abs error): (i) mean(W) <= 1.05 min(B), (ii) |slope(W)| |W| <= 0.05 min(B)
  W = final 20% of epochs (>= 30), B = epochs >= ceil(0.1 E) (same windows).
B3-frozen: v4 criterion is primary; B3-ft: self-vision criterion is primary.
Writes JSON + learning-curve figure (all epochs; train self_vision per pixel,
feature_prediction_self / _other; eval-group self_vision every test_interval)
to data/result/v10_swap/b3/.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR, git_hash  # noqa

import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

STEPS = 100  # prediction steps per episode summed in the logged self_vision loss
PRIMARY = {'v10_b3_frozen': 'v4_fp_other', 'v10_b3_ft': 'self_vision'}


def load(exp, seed, mode, name):
    p = os.path.join(PI_DIR, 'data', 'result', exp, str(seed), 'log', mode, f'{name}.log')
    if not os.path.exists(p):
        return None, None
    xs, ys = [], []
    for line in open(p):
        if line.strip():
            e, v = line.split(',')
            xs.append(int(e))
            ys.append(float(v))
    return np.array(xs), np.array(ys)


def criterion(xs, ys, level, drift_thr):
    E = len(ys)
    w = min(E, max(30, int(round(0.2 * E))))
    burn = -(-E // 10)
    B = ys[burn:]
    Wx, Wy = xs[E - w:], ys[E - w:]
    vmin = float(B.min())
    mW = float(Wy.mean())
    slope = float(np.polyfit(Wx, Wy, 1)[0])
    drift = abs(slope) * float(Wx[-1] - Wx[0])
    trans = [int(x) for x, y in zip(xs, ys) if y < 1.3 * vmin]
    return {'E': int(E), 'W': [int(Wx[0]), int(Wx[-1])], 'min_B': vmin, 'mean_W': mW,
            'ratio_level': mW / vmin, 'net_drift_over_W': drift, 'ratio_drift': drift / vmin,
            'signed_slope': slope, 'cond_level': bool(mW <= level * vmin),
            'cond_drift': bool(drift <= drift_thr * vmin),
            'converged': bool(mW <= level * vmin and drift <= drift_thr * vmin),
            'transition_epoch': trans[0] if trans else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exps', nargs='+', default=['v10_b3_frozen', 'v10_b3_ft'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--tag', default='b3')
    args = ap.parse_args()
    res = {'git_commit': git_hash(), 'runs': {}}
    curves = {}
    for exp in args.exps:
        for s in args.seeds:
            x, fpo = load(exp, s, 'train', 'feature_prediction_other')
            if x is None:
                continue
            _, fps = load(exp, s, 'train', 'feature_prediction_self')
            xv, sv = load(exp, s, 'train', 'self_vision')
            xe, sve = load(exp, s, 'eval', 'self_vision')
            r = {'v4_fp_other': criterion(x, fpo, 1.25, 0.10),
                 'self_vision': criterion(xv, sv / STEPS, 1.05, 0.05),
                 'primary': PRIMARY.get(exp, 'v4_fp_other')}
            r['converged_primary'] = r[r['primary']]['converged']
            res['runs'][f'{exp}/s{s}'] = r
            curves[(exp, s)] = dict(x=x, fpo=fpo, fps=fps, xv=xv, sv=sv / STEPS, xe=xe,
                                    sve=None if sve is None else sve / STEPS)
    out = os.path.join(OUT_ROOT, 'b3')
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f'convergence_{args.tag}.json'), 'w'), indent=2)
    for k, r in res['runs'].items():
        a, b = r['v4_fp_other'], r['self_vision']
        print(f"{k:22s} E={a['E']} | v4 fp_other: min(B) {a['min_B']:.3f} level {a['ratio_level']:.3f} drift {a['ratio_drift']:.3f} "
              f"trans@{a['transition_epoch']} -> {'CONV' if a['converged'] else 'not'} | self_vision: min(B) {b['min_B']:.5f} "
              f"level {b['ratio_level']:.4f} drift {b['ratio_drift']:.4f} (slope {b['signed_slope']:+.2e}) -> {'CONV' if b['converged'] else 'not'} "
              f"| primary {r['primary']}: {'CONV' if r['converged_primary'] else 'NOT'}")

    if not curves:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)
    cols = {0: '#4C78A8', 1: '#F58518', 2: '#54A24B'}
    for (exp, s), c in curves.items():
        ls = '-' if exp.endswith('frozen') else '--'
        lab = f'{exp.replace("v10_b3_", "B3-")} s{s}'
        axes[0].plot(c['xv'], c['sv'], ls, color=cols[s], lw=1.2, label=lab)
        axes[1].plot(c['x'], c['fpo'], ls, color=cols[s], lw=1.2, label=lab)
        axes[2].plot(c['x'], c['fps'], ls, color=cols[s], lw=1.2, label=lab)
    axes[0].set_ylabel('self-vision L1 per pixel (train)')
    axes[1].set_ylabel('feature_prediction_other (train)')
    axes[2].set_ylabel('feature_prediction_self (train)')
    for ax in axes:
        ax.set_xlabel('epoch')
        ax.grid(color='0.92')
    axes[0].legend(fontsize=8)
    fig.suptitle(f'v10 {args.tag}: learning curves (solid = B3-frozen, dashed = B3-ft)')
    fig.savefig(os.path.join(out, f'learning_curves_{args.tag}.png'), dpi=110)


if __name__ == '__main__':
    main()
