"""
v11 non-pre-registered check: why Fig.5 fails (docs/v11_experiment_log.md Sec.7.41;
method recorded before computing). Common test (v11_s1_random test), ORIGINAL-layout
episodes only, every 5th step. Models: the 6 v11 runs (mse ep200; encoders = l1 ep200)
and v4's paper encoder (exp1_l1_1000 ep200, read-only from the v10 tree).
  1. linear CKA(sv_enc, ov_enc) and mean cosine similarity per frame;
  2. Ridge(1) read-out (episode GroupKFold 5) of A-1's view and of A-2's view (each as
     its top-50 pixel PCs) from ov_enc and from sv_enc; R^2 = 1 - SSE/SST over the 50 PCs;
  3. h1 (after input frame t) -> A-1's current view v_t and next view v_{t+1} (top-50 PCs)
     and -> A-1 position; the paper model gets A-1's motion as its process-1 input.
Writes JSON under data/result/v11/main/fig5_reason/.
"""

import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v11 import DATA, RES, load_model  # noqa
import util  # noqa

STRIDE = 5
V4_ROOT = '/work/data/result'


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def pcs(V, k=50):
    V = V.reshape(len(V), -1).astype(np.float32) / 255.0
    V = V - V.mean(0)
    _, _, vt = np.linalg.svd(V, full_matrices=False)
    return V @ vt[:k].T


def cv_r2(X, Y, g):
    Yh = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y, g):
        Yh[te] = Ridge(alpha=1.0).fit(X[tr], Y[tr]).predict(X[te])
    return float(1 - ((Y - Yh) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


@torch.no_grad()
def collect(model, p_mask, sv, sm, dev, paper):
    """Returns sv_enc, ov_enc (unmasked, on frames t = 0..99 step STRIDE) and h1 after frame t."""
    N = len(sv)
    SE, OE, H = [], [], []
    for b0 in range(0, N, 50):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(sv[b0:b0 + 50]))).to(dev)
        m = torch.from_numpy(sm[b0:b0 + 50]).float().to(dev)
        B = x.shape[0]
        util.seed_all(0)
        model.init_state(B)
        se, oe, hh = [], [], []
        for t in range(100):
            pm = 0 if t == 0 else p_mask
            inp = {'self_vision': x[:, t]}
            if paper:
                inp['self_motion'] = m[:, t]
            model(inp, pm, pm)
            if t % STRIDE == 0:
                se.append(model.self_vision_encoder_module(x[:, t]).cpu().numpy())
                oe.append(model.other_vision_encoder_module(x[:, t]).cpu().numpy())
                hh.append(model.superposition_module.state['self'][0].cpu().numpy())
        SE.append(np.stack(se, 1)); OE.append(np.stack(oe, 1)); H.append(np.stack(hh, 1))
    f = lambda L: np.concatenate(L).reshape(-1, L[0].shape[-1])  # noqa
    return f(SE), f(OE), f(H)


def main():
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    g = h5py.File(os.path.join(DATA, 'v11_s1_random', 'data.h5'), 'r')['test']
    keep = np.where(g['layout'][()] == 0)[0]
    sv = g['self_vision'][()][keep]
    ov = g['other_vision'][()][keep]
    sm = g['self_motion'][()][keep]
    sp = g['self_position'][()][keep]
    ts = np.arange(0, 100, STRIDE)
    grp = np.repeat(np.arange(len(keep))[:, None], len(ts), 1).reshape(-1)
    fr = lambda a: a.reshape((-1,) + a.shape[2:])  # noqa
    A1_now = pcs(fr(sv[:, ts])); A2_now = pcs(fr(ov[:, ts])); A1_next = pcs(fr(sv[:, ts + 1]))
    pos = sp[:, ts].reshape(-1, 2)
    runs = [(f'v11_{c}_s{s}', f'v11_s1_{c}_l1_s{s}', s, 200, None, False) for c in ('random', 'mixed') for s in (0, 1, 2)]
    runs.append(('v4_paper_exp1_l1_1000', 'exp1_l1_1000', 0, 200, V4_ROOT, True))
    res = {'n_episodes_original_layout': int(len(keep))}
    for name, exp, s, e, root, paper in runs:
        model, p_mask = load_model(exp, s, e, dev, ckpt_root=root)
        SE, OE, H = collect(model, p_mask, sv, sm, dev, paper)
        cos = (SE * OE).sum(1) / np.maximum(np.linalg.norm(SE, axis=1) * np.linalg.norm(OE, axis=1), 1e-9)
        r = {'cka_sv_ov': cka(SE, OE), 'cos_mean': float(cos.mean()),
             'ov_to_A1view': cv_r2(OE, A1_now, grp), 'ov_to_A2view': cv_r2(OE, A2_now, grp),
             'sv_to_A1view': cv_r2(SE, A1_now, grp), 'sv_to_A2view': cv_r2(SE, A2_now, grp),
             'h1_to_A1view_now': cv_r2(H, A1_now, grp), 'h1_to_A1view_next': cv_r2(H, A1_next, grp),
             'h1_to_A1pos': cv_r2(H, pos, grp)}
        res[name] = r
        print(name, json.dumps({k: round(v, 3) for k, v in r.items()}), flush=True)
    out = os.path.join(RES, 'v11', 'main', 'fig5_reason')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'fig5_reason.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
