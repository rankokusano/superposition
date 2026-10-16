"""
v11 learning curves (docs/v11_experiment_log.md Sec.5.2/5.3): train every
epoch, eval (train split, unshuffled) and test every test_interval, for
self_vision and the detached feature-prediction monitors. Values are the
logger's per-batch means of the per-episode sums over 100 steps (MSE stage:
mse_loss). Reads only; writes the figure and a JSON of the curves under
data/result/v11/<sub>/.
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
KEYS = ('self_vision', 'feature_prediction_self', 'feature_prediction_other')


def read(path):
    if not os.path.exists(path):
        return np.zeros((0, 2))
    a = np.loadtxt(path, delimiter=',', ndmin=2)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exps', nargs='+', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sub', default='trial')
    ap.add_argument('--name', default='trial_curves')
    args = ap.parse_args()
    out = os.path.join(RES, 'v11', args.sub)
    os.makedirs(out, exist_ok=True)
    curves = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for exp in args.exps:
        curves[exp] = {}
        for j, k in enumerate(KEYS):
            for mode, ls in (('train', '-'), ('eval', 'o'), ('test', 's')):
                a = read(os.path.join(RES, exp, str(args.seed), 'log', mode, k + '.log'))
                curves[exp][f'{mode}/{k}'] = a.tolist()
                if len(a):
                    kw = dict(label=f'{exp} {mode}')
                    if mode == 'train':
                        axes[j].plot(a[:, 0], a[:, 1], ls, **kw)
                    else:
                        axes[j].plot(a[:, 0], a[:, 1], ls, ms=4, mfc='none', **kw)
            axes[j].set_title(k)
            axes[j].set_xlabel('epoch')
            axes[j].set_yscale('log')
    axes[0].legend(fontsize=6)
    fig.suptitle(f'v11 learning curves (seed {args.seed}); per-episode sum over 100 steps of the loss')
    fig.tight_layout()
    fig.savefig(os.path.join(out, f'{args.name}_s{args.seed}.png'), dpi=130)
    with open(os.path.join(out, f'{args.name}_s{args.seed}.json'), 'w') as f:
        json.dump(curves, f)
    for exp in args.exps:
        tr = np.array(curves[exp]['train/self_vision'])
        if len(tr):
            print(exp, 'train self_vision first/min/last', tr[0, 1], tr[:, 1].min(), tr[-1, 1], 'epochs', len(tr))


if __name__ == '__main__':
    main()
