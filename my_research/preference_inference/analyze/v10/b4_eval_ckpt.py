"""
v10 B4, one model checkpoint on the v10_swapmix_r2 test split
(docs/v10_experiment_log.md Sec.5.6, 5.9 (2), 5.17, 5.22 (0), 5.30):
  - layout read-out (logistic, standardised, C=1.0, 5-fold by episode, folds
    RandomState(0)) at t = 1,5,10,20,50,100 from h1, from the model's own
    probe-Q and from its own Encoder output (sv_enc, before masking);
    per-episode correctness saved for the bootstrap
  - R^2 of "distance to red" and "distance to top-left (-9,9)" from h1
    (Ridge alpha=1, in-sample, all frames): all / original / swapped episodes
  - pixel-wise self-vision prediction error per layout (saved prediction/truth)
Input: a test_v10.py output dir with save/saved.h5 (`test` group: state,
self_position, self_vision). Output: JSON + NPZ (per-episode correctness).
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import PI_DIR, git_hash  # noqa
from layout_readout_v10 import TIMES, folds, load_model  # noqa
sys.path.insert(0, PI_DIR)
import util  # noqa

TL = np.array([-9.0, 9.0])
RED = {0: np.array([-9.0, 9.0]), 1: np.array([-9.0, -9.0])}


def cv_pred(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in folds(groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=10000))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return (pred == y).astype(np.int8)


def r2_fit(X, Y):
    return float(r2_score(Y, Ridge(alpha=1.0).fit(X, Y).predict(X)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--dataset', default='v10_swapmix_r2')
    args = ap.parse_args()
    dev = torch.device(args.device)
    m = load_model(args.exp_config, args.seed, args.epoch, dev)

    with h5py.File(os.path.join(args.run_dir, 'save', 'saved.h5'), 'r') as f:
        g = f[f'{args.epoch:05d}/test']
        h1 = g['state/self/hidden'][()]                 # (N, 100, 128)
        sp = g['self_position/input'][()]               # (N, 100, 2)
        pred = g['self_vision/prediction'][()]
        truth = g['self_vision/truth'][()]
    with h5py.File(os.path.join(PI_DIR, 'data', 'data', args.dataset, 'data.h5'), 'r') as f:
        lay = f['test/layout'][()].astype(int)
        frames = f['test/self_vision'][:, [t - 1 for t in TIMES]]
    N = len(lay)
    assert h1.shape[0] == N
    x = torch.from_numpy(util.scale_vision(util.transpose_vision(frames.reshape(-1, *frames.shape[2:]), dim=4))).to(dev)
    with torch.no_grad():
        pq = m.compute_probe_q((x + 1) / 2).cpu().numpy().reshape(N, len(TIMES), -1)
        se = m.self_vision_encoder_module(x).cpu().numpy().reshape(N, len(TIMES), -1)
    feats = {'h1': h1[:, [t - 1 for t in TIMES]], 'probe_q': pq, 'sv_enc': se}
    groups = np.arange(N)
    res = {'dataset': args.dataset, 'exp_config': args.exp_config, 'seed': args.seed, 'epoch': args.epoch, 'n_episodes': int(N),
           'n_original': int((lay == 0).sum()), 'n_swap': int((lay == 1).sum()), 'times': TIMES,
           'git_commit': git_hash(), 'readout_accuracy': {}, 'distance_r2': {}, 'vision_error': {}}
    correct = {}
    for k, F in feats.items():
        res['readout_accuracy'][k] = {}
        for i, t in enumerate(TIMES):
            c = cv_pred(F[:, i], lay, groups)
            correct[f'{k}_t{t}'] = c
            res['readout_accuracy'][k][str(t)] = float(c.mean())

    red = np.stack([RED[c] for c in lay])[:, None, :]
    d_red = np.linalg.norm(sp - red, axis=2)
    d_tl = np.linalg.norm(sp - TL, axis=2)
    for name, mask in (('all', np.ones(N, bool)), ('original', lay == 0), ('swap_rg', lay == 1)):
        X = h1[mask].reshape(-1, h1.shape[2])
        res['distance_r2'][name] = {'dist_red': r2_fit(X, d_red[mask].reshape(-1, 1)),
                                    'dist_top_left': r2_fit(X, d_tl[mask].reshape(-1, 1)),
                                    'corr_dist_red_top_left': float(np.corrcoef(d_red[mask].ravel(), d_tl[mask].ravel())[0, 1])}
    ep_err = np.abs(pred - truth).mean(axis=(1, 2, 3, 4))
    for name, mask in (('all', np.ones(N, bool)), ('original', lay == 0), ('swap_rg', lay == 1)):
        res['vision_error'][name] = float(ep_err[mask].mean())
    json.dump(res, open(args.out_prefix + '.json', 'w'), indent=2)
    np.savez(args.out_prefix + '_correct.npz', layout=lay, ep_vision_error=ep_err, **correct)
    print(json.dumps({'readout_t20': {k: res['readout_accuracy'][k]['20'] for k in feats},
                      'distance_r2_swap': res['distance_r2']['swap_rg']}))


if __name__ == '__main__':
    main()
