"""
Paper round 14 rev4, Sec.1.6 (docs/paper_extra_log.md Sec.38.4): timing of the Sec.2.5 procedure on ARTIFICIAL arrays
(same shapes as stage C: 4 pairs x 3 models (50%, none, pre-training) x 5 checkpoints x 1200 episodes). No stage-C
result is computed or read. CPU only, one thread per process (OMP/MKL/OPENBLAS = 1).
  folds: 5, stratified by layout x preference, assigned once (seed 0); per-fold standardisation on the training side.
  two-stage bootstrap replicate: pairs resampled (4 of 4); ONE episode resample (1200, original indices, duplicates
  stay in their original fold) shared by all selected pairs, models and checkpoints.
  (L1) q~_e regressions: per model x checkpoint, q_e standardised on the resample, OLS stage 1 (1200: pref, red, y0 A2,
       y0 A1) and stage 2 (direction present: + dir); main forms only, plus a variant timing with the sensitivity forms.
  (L2) Q2 (8-dim) read-outs refitted in every replicate: basic (Q only) and adjusted (baseline covariates vs
       covariates + Q; direction present), L2 logistic C = 1.0, 5 folds; BA (mean over the 4 layout x preference
       cells) and log loss; generalisation fits timed separately.
  (L3) light procedure: read-outs (64 / 128 dims) fitted once per fold on the original data, validation probabilities
       kept; each replicate only re-scores the resampled predictions.
Signal strength of the artificial features: weak and strong (separable) to bracket the fit time.
-> data/result/paper_extra/StageC/timing/summary.json
"""

import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa
from sklearn.linear_model import LogisticRegression  # noqa

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'result',
                   'paper_extra', 'StageC', 'timing')
N, P, M, K, F = 1200, 4, 3, 5, 5
REPS = 10


def make(sig, seed=1):
    rng = np.random.default_rng(seed)
    lay = np.repeat([0, 1, 0, 1], 300)
    liked = np.repeat([1, 1, 0, 0], 300)
    s_red = np.where(lay == 0, 1.0, -1.0)
    s_pref = np.where(liked == 1, s_red, -s_red)
    y2, y1 = rng.uniform(-10, 10, N), rng.uniform(-10, 10, N)
    p = 0.8 * s_pref - 0.2 * y2 + rng.normal(0, 2.0, N)
    keep = np.abs(p) >= 0.3
    keep[np.argsort(np.abs(p))[:112]] = False
    s_dir = np.sign(p)
    frac = 1 / (1 + np.exp(-p))
    cell = liked * 2 + lay
    fold = np.empty(N, int)
    r0 = np.random.default_rng(0)
    for c in range(4):
        idx = np.where(cell == c)[0]
        r0.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % F
    lab = dict(lay=lay, s_pref=s_pref, s_red=s_red, y2=y2, y1=y1, s_dir=s_dir, keep=keep, frac=frac, cell=cell, fold=fold)
    q = rng.normal(size=(P, M, K, N)) + 0.3 * s_pref
    def feat(d):  # noqa
        w = rng.normal(size=d)
        return rng.normal(size=(P, M, K, N, d)) + sig * s_pref[:, None] * w + 0.5 * s_dir[:, None] * rng.normal(size=d)
    return lab, q, feat(8), feat(128), feat(64)[0, 0, 0]


def ols(X, Y):
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def l1_rep(lab, q, pi, ei, forms=('s1', 's2')):
    L = {k: v[ei] for k, v in lab.items()}
    base = [np.ones(N), L['s_pref'], L['s_red'], L['y2'], L['y1']]
    out = []
    for j in pi:
        Y = q[j].reshape(M * K, N)[:, ei].T                      # (N, M*K)
        Y = (Y - Y.mean(0)) / Y.std(0)
        res = {}
        if 's1' in forms:
            res['s1'] = ols(np.column_stack(base), Y)[1]
        kp = L['keep']
        if 's2' in forms:
            res['s2'] = ols(np.column_stack(base + [L['s_dir']])[kp], Y[kp])[1]
        for extra in forms[2:]:                                      # sensitivity forms (timing only)
            res[extra] = ols(np.column_stack(base + [L['s_dir'], L['frac']] + [np.random.rand(N)] * 2)[kp], Y[kp])[1]
        out.append(res)
    return out


def ba(prob, y, cell):
    hit = (prob > 0.5) == (y > 0)
    return np.mean([hit[cell == c].mean() for c in range(4) if (cell == c).any()])


def logloss(prob, y):
    p = np.clip(prob, 1e-12, 1 - 1e-12)
    return float(-np.mean(np.where(y > 0, np.log(p), np.log(1 - p))))


def fit_pred(Xtr, ytr, Xva):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    m = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr - mu) / sd, ytr)
    return m.predict_proba((Xva - mu) / sd)[:, 1]


def l2_rep(lab, Q, pi, ei, gen=False):
    L = {k: v[ei] for k, v in lab.items()}
    y = (L['s_pref'] > 0).astype(int)
    cov = np.column_stack([L['s_dir'], L['frac'], L['y2'], L['y1'], L['lay']])
    kp = L['keep']
    # baseline (covariates only): shared by all models in the replicate
    pb = np.zeros(N)
    for f in range(F):
        tr, va = (L['fold'] != f) & kp, (L['fold'] == f) & kp
        pb[va] = fit_pred(cov[tr], y[tr], cov[va])
    llb = logloss(pb[kp], y[kp])
    out = []
    for j in pi:
        for m in range(M):
            for k in range(K):
                X = Q[j, m, k][ei]
                pq, pa = np.zeros(N), np.zeros(N)
                for f in range(F):
                    tr, va = L['fold'] != f, L['fold'] == f
                    pq[va] = fit_pred(X[tr], y[tr], X[va])
                    tr2, va2 = tr & kp, va & kp
                    pa[va2] = fit_pred(np.hstack([cov, X])[tr2], y[tr2], np.hstack([cov, X])[va2])
                r = (ba(pq, y, L['cell']), logloss(pq, y), llb - logloss(pa[kp], y[kp]))
                if gen:
                    agree, dis = kp & (L['s_dir'] == L['s_pref']), kp & (L['s_dir'] != L['s_pref'])
                    r += (((fit_pred(X[agree], y[agree], X[dis]) > .5) == y[dis]).mean(),
                          ((fit_pred(X[L['lay'] == 0], y[L['lay'] == 0], X[L['lay'] == 1]) > .5) == y[L['lay'] == 1]).mean(),
                          ((fit_pred(X[L['lay'] == 1], y[L['lay'] == 1], X[L['lay'] == 0]) > .5) == y[L['lay'] == 0]).mean())
                out.append(r)
    return out


def l3_prefit(lab, R):
    """one-time fits on the original data: validation probabilities per model x checkpoint (basic, adjusted)."""
    y = (lab['s_pref'] > 0).astype(int)
    cov = np.column_stack([lab['s_dir'], lab['frac'], lab['y2'], lab['y1'], lab['lay']])
    kp = lab['keep']
    shp = R.shape[:-2] if R.ndim == 5 else ()
    PQ, PA = np.zeros(shp + (N,)), np.zeros(shp + (N,))
    for ix in np.ndindex(*shp):
        X = R[ix]
        for f in range(F):
            tr, va = lab['fold'] != f, lab['fold'] == f
            PQ[ix][va] = fit_pred(X[tr], y[tr], X[va])
            tr2, va2 = tr & kp, va & kp
            PA[ix][va2] = fit_pred(np.hstack([cov, X])[tr2], y[tr2], np.hstack([cov, X])[va2])
    return PQ, PA


def l3_rep(lab, PQ, PA, pb, pi, ei):
    y = (lab['s_pref'][ei] > 0).astype(int)
    kp, cell = lab['keep'][ei], lab['cell'][ei]
    llb = logloss(pb[ei][kp], y[kp])
    out = []
    for j in pi:
        for m in range(M):
            for k in range(K):
                pq, pa = PQ[j, m, k][ei], PA[j, m, k][ei]
                out.append((ba(pq, y, cell), logloss(pq, y), llb - logloss(pa[kp], y[kp])))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {'reps_timed': REPS, 'threads_per_process': os.environ.get('OMP_NUM_THREADS'), 'note': 'artificial arrays only'}
    for sig in (0.15, 1.0):
        lab, q, Q8, R128, O64 = make(sig)
        rng = np.random.default_rng(0)
        draws = [(rng.integers(0, P, P), rng.integers(0, N, N)) for _ in range(REPS)]
        t = time.time()
        for pi, ei in draws:
            l1_rep(lab, q, pi, ei)
        R[f'L1_main_sig{sig}_s_per_rep'] = (time.time() - t) / REPS
        t = time.time()
        for pi, ei in draws:
            l1_rep(lab, q, pi, ei, forms=('s1', 's2', 'v', 'i', 'ii', 'iii', 'iv', 'n1088', 'raw'))
        R[f'L1_with_sensitivity_sig{sig}_s_per_rep'] = (time.time() - t) / REPS
        t = time.time()
        for pi, ei in draws:
            l2_rep(lab, Q8, pi, ei)
        R[f'L2_sig{sig}_s_per_rep'] = (time.time() - t) / REPS
        t = time.time()
        for pi, ei in draws[:3]:
            l2_rep(lab, Q8, pi, ei, gen=True)
        R[f'L2_with_generalisation_sig{sig}_s_per_rep'] = (time.time() - t) / 3
        t = time.time()
        PQ, PA = l3_prefit(lab, R128)
        R[f'L3_prefit_128dim_one_representation_sig{sig}_s'] = time.time() - t
        t = time.time()
        pb = np.full(N, 0.5)
        for pi, ei in draws:
            l3_rep(lab, PQ, PA, pb, pi, ei)
        R[f'L3_rep_one_representation_sig{sig}_s_per_rep'] = (time.time() - t) / REPS
        t = time.time()
        PQo, PAo = l3_prefit(lab, O64[None, None, None].repeat(1, 0))
        R[f'L3_prefit_64dim_one_model_ckpt_sig{sig}_s'] = time.time() - t
        print(sig, {k: round(v, 3) for k, v in R.items() if isinstance(v, float)}, flush=True)
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
