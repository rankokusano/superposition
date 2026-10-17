"""
v11 convergence (docs/v11_experiment_log.md Sec.5.3, 7.13). Exactly the
registered windows (not check_convergence_v10.py's 10%-burn-in variant):
  W = final 20% of the epochs, B = epochs >= 40.
  self-vision criterion (primary, mse stage; reference, l1 stage) on the
  train-log self_vision / 100 steps:  mean(W) <= 1.05 min(B) and
  |OLS slope(W)| * |W| <= 0.05 min(B)   (|W| = epoch span of W)
  v4 criterion (reference) on train feature_prediction_other: 1.25 / 0.10.
Writes JSON (+ all-seed learning-curve figure) under data/result/v11/<sub>/.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(_PI_DIR, 'data', 'result')


def load(exp, seed, mode, name):
    p = os.path.join(RES, exp, str(seed), 'log', mode, f'{name}.log')
    if not os.path.exists(p):
        return None
    a = np.loadtxt(p, delimiter=',', ndmin=2)
    return a[:, 0].astype(int), a[:, 1]


def criterion(xs, ys, level, drift_thr, burn=40):
    E = len(ys)
    w = int(round(0.2 * E))
    B = ys[xs >= burn]
    Wx, Wy = xs[E - w:], ys[E - w:]
    vmin, mW = float(B.min()), float(Wy.mean())
    slope = float(np.polyfit(Wx, Wy, 1)[0])
    drift = abs(slope) * float(Wx[-1] - Wx[0])
    return {'E': int(E), 'W': [int(Wx[0]), int(Wx[-1])], 'B_from': burn, 'min_B': vmin, 'mean_W': mW,
            'ratio_level': mW / vmin, 'ratio_drift': drift / vmin, 'signed_slope': slope,
            'converged': bool(mW <= level * vmin and drift <= drift_thr * vmin)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['mse', 'l1'], default='mse')
    ap.add_argument('--exp_fmt', default=None, help="default v11_s1_{c}_mse / v11_s1_{c}_l1_s{s}")
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--sub', default='main/convergence')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    res = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for c, col in (('random', 'tab:blue'), ('mixed', 'tab:red')):
        for s in args.seeds:
            exp = (args.exp_fmt or ('v11_s1_{c}_mse' if args.stage == 'mse' else 'v11_s1_{c}_l1_s{s}')).format(c=c, s=s)
            sv = load(exp, s, 'train', 'self_vision')
            fp = load(exp, s, 'train', 'feature_prediction_other')
            if sv is None:
                continue
            r = {'exp': exp, 'last_epoch': int(sv[0][-1])}
            if sv[0][-1] < 50:
                r['note'] = 'too few epochs for the criterion (B = epochs >= 40)'
            else:
                r['self_vision_per_pixel'] = criterion(sv[0], sv[1] / 100.0, 1.05, 0.05)
            if fp is not None and 'note' not in r:
                r['v4_fp_other'] = criterion(fp[0], fp[1], 1.25, 0.10)
            res[f'{c}_s{s}'] = r
            ls = ['-', '--', ':'][s % 3]
            axes[0].plot(sv[0], sv[1] / 100.0, ls, color=col, label=f'{c} s{s}')
            if fp is not None:
                axes[2].plot(fp[0], fp[1], ls, color=col)
            ev = load(exp, s, 'eval', 'self_vision')
            if ev is not None:
                axes[1].plot(ev[0], ev[1] / 100.0, ls, color=col, marker='o', ms=3)
    axes[0].set_title('train self_vision / 100 steps'); axes[1].set_title('eval (train split) self_vision / 100')
    axes[2].set_title('train feature_prediction_other (v4 criterion)')
    for a in axes:
        a.set_xlabel('epoch'); a.set_yscale('log')
    axes[0].legend(fontsize=7)
    fig.suptitle(f'v11 main run, {args.stage} stage: learning curves (all seeds)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'curves_{args.stage}.png'), dpi=120)
    with open(os.path.join(out, f'convergence_{args.stage}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    for k, r in res.items():
        if 'note' in r:
            print(k, r['last_epoch'], r['note'])
            continue
        print(k, r['last_epoch'], 'self-vision converged', r['self_vision_per_pixel']['converged'],
              'ratios %.3f %.3f' % (r['self_vision_per_pixel']['ratio_level'], r['self_vision_per_pixel']['ratio_drift']),
              '| v4', r.get('v4_fp_other', {}).get('converged'))


if __name__ == '__main__':
    main()
