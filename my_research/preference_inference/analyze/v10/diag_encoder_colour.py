"""
v10 stage A, EXPLORATORY diagnostic (not pre-registered; added 2026-10-15
after seeing the prediction images): does the frozen visual Encoder keep the
red/green identity, or only where landmarks are?

For the self/other vision encoders of a shared-module checkpoint, on the
paired viz20 episodes (r2_a1random_a2rl_viz20 vs v10_swap_r2_viz20; A-1 at
the same position in each paired frame; only red/green differ):
  - paired distance: mean ||enc(A0 frame) - enc(A1 frame)|| over paired frames,
    relative to the mean distance between two different frames of the same
    layout (random pairs);
  - layout decoding: logistic regression (standardised, C=1.0, 5-fold by
    episode, seed 0) of layout from the encoder output, per frame;
    references: from the raw pixels, and from the 8-dim probe-Q.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR, git_hash  # noqa
sys.path.insert(0, PI_DIR)
import model as models  # noqa
import util  # noqa
from constants import MODEL_DIR_BASE, RESULT_DIR  # noqa


class A:
    pass


def load_model(exp, seed, epoch, device):
    args = A()
    args.exp_config, args.seed = exp, seed
    exp_config = util.gen_exp_config(args)
    m = getattr(models, exp_config.model.name)(util.gen_model_config(exp_config)).to(device)
    ck = os.path.join(PI_DIR, RESULT_DIR, exp, str(seed), MODEL_DIR_BASE, f'{epoch:05d}.pth')
    m.load_state_dict(torch.load(ck, map_location='cpu')['model'])
    return m.eval()


def frames(path):
    with h5py.File(path, 'r') as f:
        v = f['train/self_vision'][()]  # (20, 101, 16, 64, 3) uint8
    n, t = v.shape[:2]
    return v.reshape(n * t, *v.shape[2:]), np.repeat(np.arange(n), t)


def decode(X, y, groups):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    acc = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf.fit(X[tr], y[tr])
        acc.append(clf.score(X[te], y[te]))
    return float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    np.random.seed(0)
    dev = torch.device(args.device)
    m = load_model(args.exp_config, args.seed, args.epoch, dev)
    v0, g0 = frames(os.path.join(PI_DIR, 'data/data/r2_a1random_a2rl_viz20/data.h5'))
    v1, g1 = frames(os.path.join(PI_DIR, 'data/data/v10_swap_r2_viz20/data.h5'))

    def enc(v):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(v, dim=4))).to(dev)
        with torch.no_grad():
            out = {'sv_enc': m.self_vision_encoder_module(x).cpu().numpy(),
                   'ov_enc': m.other_vision_encoder_module(x).cpu().numpy()}
            if hasattr(m, 'compute_probe_q'):
                out['probe_q'] = m.compute_probe_q((x + 1) / 2).cpu().numpy()
        return out

    e0, e1 = enc(v0), enc(v1)
    e0['pixels'] = v0.reshape(len(v0), -1).astype(np.float32) / 255
    e1['pixels'] = v1.reshape(len(v1), -1).astype(np.float32) / 255
    y = np.r_[np.zeros(len(v0)), np.ones(len(v1))]
    groups = np.r_[g0, g1]  # paired episodes share a group -> never split across folds
    res = {'exp_config': args.exp_config, 'seed': args.seed, 'epoch': args.epoch,
           'git_commit': git_hash(), 'n_frames_per_layout': int(len(v0)),
           'note': 'exploratory, not pre-registered', 'features': {}}
    for k in e0:
        a, b = e0[k], e1[k]
        paired = np.linalg.norm(a - b, axis=1).mean()
        idx = np.random.permutation(len(a))
        within = np.linalg.norm(a - a[idx], axis=1)[idx != np.arange(len(a))].mean()
        res['features'][k] = {
            'dim': int(a.shape[1]),
            'paired_distance_A0_vs_A1': float(paired),
            'random_pair_distance_within_A0': float(within),
            'ratio_paired_over_random': float(paired / within),
            'layout_decoding_accuracy_5fold_by_episode': decode(np.r_[a, b], y, groups),
        }
    out = os.path.join(OUT_ROOT, 'stageA', 'diag', f'encoder_colour_{args.exp_config}_s{args.seed}_ep{args.epoch}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
