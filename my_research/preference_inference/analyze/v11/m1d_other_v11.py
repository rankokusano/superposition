"""
v11 stage-2 M1(d)-other (docs/v11_experiment_log.md Sec.7.52). For one stage-2 run and G
checkpoint, Q2 = generated / P_A2 (D6) / zero, on v11_s2t_green (liked Green) and v11_s2t_red
(liked Red): s_X = L1(pred, v_{t+1}) - L1(pred, cf_X) (prediction and views in [0, 1]) on frames
where A-2 is awake, moved, and its step is >= 45 deg away from (ref_X - p2_t); S_X = episode
means; c_cf_other = S_liked - mean of the other three. Masking / seeding as eval_s2_v11.py.
Writes data/result/v11/stage2/m1d_other/<tag>.json and per-episode S (npz).
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import load_model  # noqa
from eval_s2_v11 import OUT, load_set, true_pleasure  # noqa
from eval_m1_v11 import refs  # noqa
import util  # noqa

COLORS = ('Red', 'Green', 'Blue', 'Cyan')
CF = os.path.join(os.path.dirname(os.path.dirname(OUT)), 'cf_other')
OUTM = os.path.join(os.path.dirname(OUT), 'm1d_other')
LIKED = {'green': 'Green', 'red': 'Red'}


def load_cf(name):
    parts = sorted(glob.glob(os.path.join(CF, f'v11_s2t_{name}_part*.h5')))
    eps, cv, va = [], [], []
    for p in parts:
        with h5py.File(p, 'r') as f:
            eps.append(f['episodes'][()]); cv.append(f['cf_vision'][()]); va.append(f['valid'][()])
    eps = np.concatenate(eps)
    o = np.argsort(eps)
    assert np.array_equal(eps[o], np.arange(len(eps)))
    return np.concatenate(cv)[o], np.concatenate(va)[o]


@torch.no_grad()
def errs(model, d, cf, dev, p2=None, batch=50):
    """-> err (N, 100), lcf (N, 100, 4)."""
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    E, LC = [], []
    util.seed_all(0)
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        model.init_state(x.shape[0])
        e, lc = [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            ov = None if p2 is None else torch.from_numpy(p2[b0:b0 + batch, t]).float().to(dev)
            pred = model({'self_vision': x[:, t]}, pm, pm, p2_override=ov)['self_vision']
            p01 = ((pred + 1) / 2).clamp(0, 1)
            e.append((p01 - (x[:, t + 1] + 1) / 2).abs().mean((1, 2, 3)).cpu().numpy())
            c = torch.from_numpy(cf[b0:b0 + batch, t]).to(dev).float().permute(0, 1, 4, 2, 3) / 255.0  # (B,4,3,16,64)
            lc.append((p01[:, None] - c).abs().mean((2, 3, 4)).cpu().numpy())
        E.append(np.stack(e, 1)); LC.append(np.stack(lc, 1))
    return np.concatenate(E), np.concatenate(LC)


def m1d_other(d, err, lcf, valid, liked):
    p = d['other_position'][:, :100]
    m = d['other_position'][:, 1:101] - p
    ok0 = (d['a2_source'][:, :100] != 3) & (np.linalg.norm(m, axis=-1) > 1e-6)
    S = {}
    for k, X in enumerate(COLORS):
        u = refs(d['layout'], X) - p
        cosang = (m * u).sum(-1) / np.maximum(np.linalg.norm(m, axis=-1) * np.linalg.norm(u, axis=-1), 1e-9)
        sel = ok0 & valid[:, :, k] & (cosang < np.cos(np.deg2rad(45)))
        sx = err - lcf[:, :, k]
        S[X] = np.array([sx[i][sel[i]].mean() if sel[i].any() else np.nan for i in range(len(p))])
    others = [X for X in COLORS if X != liked]
    c = S[liked] - sum(S[X] for X in others) / 3
    return S, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    args = ap.parse_args()
    os.makedirs(OUTM, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in ('green', 'red')}
    PA2 = {k: true_pleasure(k, D[k]) for k in D}
    CFV = {k: load_cf(k) for k in D}
    for spec in args.runs:
        st, c, s = spec.split(':')
        for e in args.epochs:
            tag = f'{st}_{c}_s{s}_ep{e}'
            model, _ = load_model(f'v11_s2_{st}_{c}_s{s}', int(s), e, dev)
            r, per = {'tag': tag}, {}
            for k in D:
                for cond, p2 in (('gen', None), ('d6', PA2[k]), ('zero', np.zeros_like(PA2[k]))):
                    err, lcf = errs(model, D[k], CFV[k][0], dev, p2)
                    S, cc = m1d_other(D[k], err, lcf, CFV[k][1], LIKED[k])
                    r[f'{k}_{cond}'] = {'c_cf_other': float(np.nanmean(cc)), 'n_ep': int(np.isfinite(cc).sum()),
                                        'S': {X: float(np.nanmean(v)) for X, v in S.items()}}
                    per[f'{k}_{cond}_c'] = cc
            json.dump(r, open(os.path.join(OUTM, f'{tag}.json'), 'w'), indent=1)
            np.savez_compressed(os.path.join(OUTM, f'{tag}_per.npz'), **per)
            print(tag, json.dumps({k: round(v['c_cf_other'], 5) for k, v in r.items() if k != 'tag'}), flush=True)


if __name__ == '__main__':
    main()
