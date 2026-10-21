"""
v12 post-hoc analysis B (docs/v12_experiment_log.md Sec.41; not pre-registered, judgements unchanged).
For a stage-2 model family (prefix) and pair, at G ep200, stage-2 protocol:
  B1/B2: A-2's 'preference' (set 1: layout; set 2: the episode's a2_liked) read from generated Q2 and from h2:
         logistic (linear) and 25-NN (non-linear), episode GroupKFold(5), frames t >= 20; position-controlled
         (poly-2 / raw positions of A-1 and A-2 alone vs + features); episode-level (time-mean features).
  B3:    G5a with the GENERATED Q2: at t = 20 take G's Q2 and keep it for K = 15 steps (process-2 =
         [0, 0, Q2(20), 1]) vs keep generating; g_other from the A-2 position read from the predicted images
         (primary read-out and the B1 background-relative read-out).
Writes data/result/v12/post/<prefix>_s<s>.json.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v12 import RES, load_v12, summ  # noqa
from eval_s2_v11 import load_set  # noqa
from eval_g5_v12 import g5a_metric, calibrate, calibrate_bg, read_a2, read_a2_bg, T0  # noqa
import util  # noqa

K = 15
LIN = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))  # noqa
KNN = lambda k=25: make_pipeline(StandardScaler(), KNeighborsClassifier(k))  # noqa


@torch.no_grad()
def collect(model, d, dev, batch=50):
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    util.seed_all(0)
    Q, H = [], []
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b0 + batch]).float().to(dev)
        model.init_state(x.shape[0])
        q, h = [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            out = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm)
            q.append(out['q2_hat'].cpu().numpy())
            h.append(model.get_state()['other'][0].cpu().numpy())
        Q.append(np.stack(q, 1)); H.append(np.stack(h, 1))
    return np.concatenate(Q), np.concatenate(H)


def cv_acc(clf_fn, X, y, g):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, g):
        pred[te] = clf_fn().fit(X[tr], y[tr]).predict(X[te])
    return float((pred == y).mean())


def readouts(F, d, target):
    N = F.shape[0]
    ts = np.arange(20, 100)
    X = F[:, ts].reshape(-1, F.shape[-1])
    y = np.repeat(target[:, None], len(ts), 1).reshape(-1)
    g = np.repeat(np.arange(N)[:, None], len(ts), 1).reshape(-1)
    pos = np.concatenate([d['self_position'][:, ts].reshape(-1, 2), d['other_position'][:, ts].reshape(-1, 2)], 1)
    P2 = PolynomialFeatures(2).fit_transform(pos)
    sub = np.arange(0, len(y), 4)    # every 4th frame keeps the k-NN tractable
    r = {'frames_linear': cv_acc(LIN, X, y, g), 'frames_knn': cv_acc(KNN, X[sub], y[sub], g[sub]),
         'pos_only_linear': cv_acc(LIN, P2, y, g), 'pos_plus_linear': cv_acc(LIN, np.concatenate([P2, X], 1), y, g),
         'pos_only_knn': cv_acc(KNN, pos[sub], y[sub], g[sub]),
         'pos_plus_knn': cv_acc(KNN, np.concatenate([pos, X], 1)[sub], y[sub], g[sub])}
    Xe = F[:, ts].mean(1)
    ge = np.arange(N)
    r['episode_linear'] = cv_acc(LIN, Xe, target, ge)
    r['episode_knn'] = cv_acc(lambda: KNN(15), Xe, target, ge)
    return r


@torch.no_grad()
def g5a_generated(model, d, dev, ab, bgc, freeze, batch=50):
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    util.seed_all(0)
    E, EB = [], []
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b0 + batch]).float().to(dev)
        sp = d['self_position'][b0:b0 + batch]
        B = x.shape[0]
        model.init_state(B)
        e, eb, qhold = [], [], None
        for t in range(T0 + K):
            pm = 0 if t == 0 else 1
            ov = None
            if freeze and t > T0:
                ov = torch.cat([torch.zeros(B, 2, device=dev), qhold, torch.ones(B, 1, device=dev)], 1)
            out = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=ov)
            if t == T0:
                qhold = out['q2_hat'].detach()
            if t >= T0:
                p01 = ((out['self_vision'] + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                e.append(read_a2(p01, sp[:, t + 1], *ab))
                eb.append(read_a2_bg(p01, sp[:, t + 1], *bgc))
        E.append(np.stack(e, 1)); EB.append(np.stack(eb, 1))
    return np.concatenate(E), np.concatenate(EB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True, help='v12_s2 or v12_s2b')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epoch', type=int, default=200)
    a = ap.parse_args()
    out = os.path.join(RES, 'v12', 'post'); os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in ('green', 'pref')}
    gg = h5py.File(D['green']['path'], 'r')['test']
    ab = calibrate(gg['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    bgc = calibrate_bg(gg['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    res = {'prefix': a.prefix, 'seed': a.seed, 'epoch': a.epoch}
    per = {}
    for st in ('green', 'pref'):
        for c in ('random', 'mixed'):
            model, _ = load_v12(f'{a.prefix}_{st}_{c}_s{a.seed}', a.seed, a.epoch, dev)
            r = {}
            for k, d in D.items():
                Q, H = collect(model, d, dev)
                target = d['layout'].astype(int) if k == 'green' else d['a2_liked'].astype(int)
                r[f'{k}_Q2'] = readouts(Q, d, target)
                r[f'{k}_h2'] = readouts(H, d, target)
                for freeze in (True, False):
                    est, estb = g5a_generated(model, d, dev, ab, bgc, freeze)
                    tag = 'freeze' if freeze else 'continue'
                    gm, _, _ = g5a_metric(est, d)
                    gb, _, _ = g5a_metric(estb, d)
                    r[f'{k}_B3_{tag}'] = {'g_other': summ(gm), 'g_other_bg': summ(gb), 'frac_read': float(np.isfinite(est[..., 0]).mean())}
                    per[(st, c, k, tag)] = (gm, gb)
            res[f'{st}_{c}'] = r
            print(st, c, json.dumps({kk: {k2: (round(v2, 3) if isinstance(v2, float) else v2['mean'] if isinstance(v2, dict) and 'mean' in v2 else v2)
                                          for k2, v2 in vv.items()} for kk, vv in r.items()}), flush=True)
    for st in ('green', 'pref'):
        for k in D:
            for tag in ('freeze', 'continue'):
                gm = per[(st, 'mixed', k, tag)][0] - per[(st, 'random', k, tag)][0]
                gb = per[(st, 'mixed', k, tag)][1] - per[(st, 'random', k, tag)][1]
                res.setdefault('paired_B3', {})[f'{st}_{k}_{tag}'] = {'g_other': summ(gm), 'g_other_bg': summ(gb)}
    json.dump(res, open(os.path.join(out, f'{a.prefix}_s{a.seed}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
