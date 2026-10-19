"""
v12 non-preregistered check A (docs/v12_experiment_log.md Sec.19): do the frozen v4 encoders (and v11's
from-scratch encoders) carry which colour is where? CPU only.
(1) layout (original / swapped) from sv_enc / ov_enc of A-1's view (logistic regression, episode
    GroupKFold(5) accuracy), references: top-50 pixel PCs of A-1's view, A-1's own pleasure P;
(2) on v11_s2t_green: unit vector from A-2 to the green reference point from ov_enc / sv_enc (Ridge,
    episode GroupKFold(5) R^2); references: A-2 position (poly-2), A-2 position x layout, A-1's own P.
"""

import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import DATA, RES  # noqa
from fig5_extra_v11 import make_ae, enc, cv_r2  # noqa
from eval_m1_v11 import refs  # noqa
from v11core.pleasure import pleasure_np  # noqa

TS = np.arange(0, 100, 5)


def frames(name):
    g = h5py.File(os.path.join(DATA, name, 'data.h5'), 'r')['test']
    N = g['layout'].shape[0]
    sv = g['self_vision'][()][:, TS].reshape(-1, 16, 64, 3)
    lay = np.repeat(g['layout'][()][:, None], len(TS), 1).reshape(-1)
    op = g['other_position'][()][:, TS].reshape(-1, 2)
    grp = np.repeat(np.arange(N)[:, None], len(TS), 1).reshape(-1)
    return sv, lay, op, grp


def acc_cv(X, y, g):
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(5).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return float((pred == y).mean())


def pcs(V, k=50):
    V = V.reshape(len(V), -1).astype(np.float32) / 255.0
    V = V - V.mean(0)
    _, _, vt = np.linalg.svd(V, full_matrices=False)
    return V @ vt[:k].T


def main():
    torch.set_num_threads(8)
    out = os.path.join(RES, 'v12', 'enc_colour'); os.makedirs(out, exist_ok=True)
    encs = [('v4_frozen (all v12 runs)', '/work/data/result/exp1_l1_1000/0/model/00200.pth')]
    for c in ('random', 'mixed'):
        for s in (0, 1, 2):
            encs.append((f'v11_{c}_s{s}', os.path.join(RES, f'v11_s1_{c}_mse', str(s), 'model', '00200.pth')))
    res = {}
    D = {k: frames(k) for k in ('v11_s1_random', 'v11_s2t_green')}
    ref = {}
    for k, (sv, lay, op, grp) in D.items():
        P = pleasure_np(sv)
        ref[k] = {'layout_from_pixels_pc50': acc_cv(pcs(sv), lay, grp), 'layout_from_own_P': acc_cv(P, lay, grp)}
    sv, lay, op, grp = D['v11_s2t_green']
    u = np.stack([refs(np.array([l]), 'Green')[0, 0] for l in lay]) - op
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    poly = PolynomialFeatures(2).fit_transform(op)
    ref['green_dir_from_A2pos_poly2'] = cv_r2(poly, u, grp)
    ref['green_dir_from_A2pos_x_layout'] = cv_r2(np.concatenate([poly, poly * lay[:, None]], 1), u, grp)
    ref['green_dir_from_own_P'] = cv_r2(pleasure_np(sv), u, grp)
    res['references'] = ref
    print('references', json.dumps(ref), flush=True)
    for name, ck in encs:
        m, _ = make_ae(ck)
        r = {}
        for k, (sv_, lay_, op_, grp_) in D.items():
            S, O = enc(m.self_vision_encoder_module, sv_), enc(m.other_vision_encoder_module, sv_)
            r[k] = {'layout_from_sv_enc': acc_cv(S, lay_, grp_), 'layout_from_ov_enc': acc_cv(O, lay_, grp_)}
            if k == 'v11_s2t_green':
                r[k]['green_dir_from_ov_enc'] = cv_r2(O, u, grp_)
                r[k]['green_dir_from_sv_enc'] = cv_r2(S, u, grp_)
                r[k]['green_dir_from_ov_enc_plus_A2pos'] = cv_r2(np.concatenate([O, poly], 1), u, grp_)
        res[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(res, open(os.path.join(out, 'enc_colour.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
