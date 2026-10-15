"""
v10 layout read-out (the B4 method, docs/v10_experiment_log.md Sec.5.6 / 5.9):
logistic regression of the episode's layout (original = 0, swap_rg = 1)
from a feature at time t. Features standardised, L2, C = 1.0 (fixed),
5-fold cross-validation split by episode group, folds from RandomState(0).
t = 1, 5, 10, 20, 50, 100 means "after the t-th input frame": h1 = state
index t-1 (saved states run t = 0..99), probe-Q / Encoder output from the
input frame t-1.

Features: h1 (state/self/hidden), probe-Q (8-dim, the model's own critic and
mu/sigma), sv_enc (self vision encoder output before masking).

Inputs: test_v10.py output dirs (saved.h5 with state for the `test` group)
for each dataset; the dataset's own data.h5 for frames and (if present)
the per-episode `layout` field. `--pair_groups` puts episode i of every
dataset in the same group (used when the datasets are episode-paired, e.g.
r2_a1random_a2rl + v10_swap_r2, whose A-1 paths are identical).
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import PI_DIR, git_hash  # noqa
sys.path.insert(0, PI_DIR)
import model as models  # noqa
import util  # noqa
from constants import MODEL_DIR_BASE, RESULT_DIR  # noqa

TIMES = [1, 5, 10, 20, 50, 100]
LAYOUT_OF = {'r2_a1random_a2rl': 0, 'v10_swap_r2': 1}


class _A:
    pass


def load_model(exp, seed, epoch, device):
    a = _A()
    a.exp_config, a.seed = exp, seed
    cfg = util.gen_exp_config(a)
    m = getattr(models, cfg.model.name)(util.gen_model_config(cfg)).to(device)
    ck = os.path.join(PI_DIR, RESULT_DIR, exp, str(seed), MODEL_DIR_BASE, f'{epoch:05d}.pth')
    m.load_state_dict(torch.load(ck, map_location='cpu')['model'])
    return m.eval()


def folds(groups, k=5, seed=0):
    ug = np.unique(groups)
    perm = np.random.RandomState(seed).permutation(ug)
    fold_of = {g: i % k for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in groups])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def cv_accuracy(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in folds(groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=10000))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float((pred == y).mean()), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--run_dirs', nargs='+', required=True, help='test_v10 out_dir per dataset')
    ap.add_argument('--mode', default='test')
    ap.add_argument('--pair_groups', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)
    m = load_model(args.exp_config, args.seed, args.epoch, dev)

    feats = {'h1': [], 'probe_q': [], 'sv_enc': []}
    ys, groups = [], []
    for di, (ds, rd) in enumerate(zip(args.datasets, args.run_dirs)):
        with h5py.File(os.path.join(rd, 'save', 'saved.h5'), 'r') as f:
            h1 = f[f'{args.epoch:05d}/{args.mode}/state/self/hidden'][()]  # (N, 100, 128)
        split = 'train' if args.mode == 'eval' else 'test'
        with h5py.File(os.path.join(PI_DIR, 'data', 'data', ds, 'data.h5'), 'r') as f:
            idx = [t - 1 for t in TIMES]
            fr = f[f'{split}/self_vision'][:, idx]  # (N, len(TIMES), 16, 64, 3)
            lay = f[f'{split}/layout'][()] if 'layout' in f[split] else \
                np.full(len(fr), LAYOUT_OF[ds], dtype=int)
        n = len(fr)
        assert h1.shape[0] == n
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(
            fr.reshape(-1, *fr.shape[2:]), dim=4))).to(dev)
        with torch.no_grad():
            pq = m.compute_probe_q((x + 1) / 2).cpu().numpy().reshape(n, len(TIMES), -1)
            se = m.self_vision_encoder_module(x).cpu().numpy().reshape(n, len(TIMES), -1)
        feats['h1'].append(h1[:, idx])
        feats['probe_q'].append(pq)
        feats['sv_enc'].append(se)
        ys.append(lay.astype(int))
        groups.append(np.arange(n) if args.pair_groups else np.arange(n) + di * 100000)
    y = np.concatenate(ys)
    groups = np.concatenate(groups)
    res = {'exp_config': args.exp_config, 'seed': args.seed, 'epoch': args.epoch,
           'datasets': args.datasets, 'mode': args.mode, 'pair_groups': args.pair_groups,
           'n_episodes': int(len(y)), 'frac_swap': float(y.mean()), 'times': TIMES,
           'method': 'StandardScaler + LogisticRegression(L2, C=1.0); 5-fold by episode group, folds RandomState(0)',
           'git_commit': git_hash(), 'accuracy': {}}
    for k, lst in feats.items():
        X = np.concatenate(lst)
        res['accuracy'][k] = {str(t): cv_accuracy(X[:, i], y, groups)[0] for i, t in enumerate(TIMES)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, 'w'), indent=2)
    print(json.dumps(res['accuracy']))


if __name__ == '__main__':
    main()
