"""
Paper round 17 (docs/paper_instructions_round17.md; docs/paper_extra_log.md Sec.41; exploratory, outside the
pre-registration). Behaviour variables of the stage-B set (vdirb, 1200 episodes, positions at frames 0..100) and the
cross-fitted regressions of G's Q2_t (8-dim, steps t = 0..99) on them. Step t sees frame t and predicts frame t + 1.
Variables per episode and step t (P2 = A-2 position, P1 = A-1 position; boundary indices clamped to frames 0..100):
  pos4   P2_t (x, y), P1_t (x, y)                      lay   layout code (stageC_analysis.LAB['lay'])
  past   for k = 1, 2, 5: d = P2_t - P2_max(t-k,0): dx, dy, cos, sin (cos = sin = 0 if |d| < 1e-6)
  fut    for k = 1, 2, 5: d = P2_min(t+k,100) - P2_t: dx, dy, cos, sin
  extra  net displacement P2_t - P2_0 (dx, dy); for Red and Green: v = corner reference point (vdir_stageA.LANDMARKS of
         the episode's layout) - P2_t: |v|, cos, sin; t; A-2 resting (a2_source_t == 3, asleep: no move this step)
Sets: F0 = round 16 (pos): degree-2 polynomial (cross terms) of pos4 + lay, OLS (round16_eval.pos_fit itself);
      F1 = poly2(pos4) + lay + past; F2 = F1 + fut; F3 = F2 + extra  (ridge);
      F4 = pos4 + lay + past + fut + extra, raw (gradient boosting); F5 = F4 + s_pref (gradient boosting).
Ridge: features standardised with the training rows; one alpha for the 8 outputs from ALPHAS by an inner
  leave-one-fold-out over the 4 training folds (summed squared error, steps 0..99); refit on the 4 folds.
Gradient boosting: sklearn HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
  min_samples_leaf=100, l2_regularization=0, early_stopping=False, random_state=0), one model per output dimension.
All sets: outer 5 folds of episodes (stageC_analysis.FOLD, as round 16), each fold predicted from the other four;
  training rows = all steps 0..99 of the training episodes. One-bit G (Q2 = v c^): the scalar v = Q2 . c^ is fitted
  and Q2_hat = v_hat c^ (F0 via pos_fit on the 8 dims is the same linear fit).
"""

import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round16_eval as R16  # noqa  (pos_fit, load_pos, WIN, CH)
import stageC_analysis as SC  # noqa
import vdir_stageA as V  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa
from sklearn.preprocessing import PolynomialFeatures  # noqa

ALPHAS = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5)
SETS = ('F0', 'F1', 'F2', 'F3', 'F4', 'F5')
HGB = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=0.0,
           early_stopping=False, random_state=0)
T = 100


def _disp(d):
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    u = np.where(n > 1e-6, d / np.maximum(n, 1e-12), 0.0)
    return np.concatenate([d, u], -1)                                  # dx, dy, cos, sin


def variables():
    d = R16.C.E.load_set('vdirb')
    g = h5py.File(d['path'], 'r')[d['split']]
    P2 = g['other_position'][()].astype(np.float64)                    # (N, 101, 2)
    P1 = g['self_position'][()].astype(np.float64)
    src2 = g['a2_source'][()]
    layout = g['layout'][()]
    lay = SC.LAB['lay'].astype(np.float64)
    assert (lay == layout).all()
    t = np.arange(T)
    pos4 = np.concatenate([P2[:, :T], P1[:, :T]], -1)
    past = np.concatenate([_disp(P2[:, t] - P2[:, np.maximum(t - k, 0)]) for k in (1, 2, 5)], -1)
    fut = np.concatenate([_disp(P2[:, np.minimum(t + k, 100)] - P2[:, t]) for k in (1, 2, 5)], -1)
    ex = [P2[:, t] - P2[:, :1]]
    for c in ('Red', 'Green'):
        ref = np.array([V.LANDMARKS[V.CODE_LAYOUT[int(L)]][c] for L in layout])[:, None]
        v = ref - P2[:, t]
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        ex += [n, v / np.maximum(n, 1e-12)]
    ex += [np.broadcast_to(t[None, :, None], (len(P2), T, 1)).astype(np.float64),
           (src2[:, :T] == 3)[..., None].astype(np.float64)]
    extra = np.concatenate(ex, -1)
    layc = np.repeat(lay[:, None, None], T, 1)
    poly = PolynomialFeatures(2, include_bias=False).fit_transform(pos4.reshape(-1, 4)).reshape(len(P2), T, -1)
    spref = np.repeat(SC.LAB['pref'][:, None, None], T, 1).astype(np.float64)
    X = {'F1': np.concatenate([poly, layc, past], -1)}
    X['F2'] = np.concatenate([X['F1'], fut], -1)
    X['F3'] = np.concatenate([X['F2'], extra], -1)
    X['F4'] = np.concatenate([pos4, layc, past, fut, extra], -1)
    X['F5'] = np.concatenate([X['F4'], spref], -1)
    info = {'n_features': {k: int(v.shape[-1]) for k, v in X.items()}, 'rest_frac_0_99': float((src2[:, :T] == 3).mean()),
            'rest_frac_0_49': float((src2[:, :50] == 3).mean())}
    return X, P2[:, :T], P1[:, :T], info


def r2(Q, Qh):
    out = {}
    for w, sl in R16.WIN.items():
        y, p = Q[:, sl].reshape(-1, Q.shape[-1]).astype(np.float64), Qh[:, sl].reshape(-1, Q.shape[-1]).astype(np.float64)
        ss, st = ((y - p) ** 2).sum(0), ((y - y.mean(0)) ** 2).sum(0)
        out[w] = {'pooled': float(1 - ss.sum() / st.sum()), 'per_dim': (1 - ss / np.maximum(st, 1e-30)).tolist()}
    return out


def _ridge_fit(X, Y, alphas):
    """-> one predictor per alpha (standardisation and Gram matrix from X)."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    ym = Y.mean(0)
    G, b = Z.T @ Z, Z.T @ (Y - ym)
    Bs = [np.linalg.solve(G + a * np.eye(Z.shape[1]), b) for a in alphas]
    return [lambda Xn, B=B: ((Xn - mu) / sd) @ B + ym for B in Bs]


def ridge_cf(X, Y):
    """X (N, T, p), Y (N, T, q) -> cross-fitted (N, T, q), chosen alpha per outer fold."""
    N, _, q = Y.shape
    out, chosen = np.zeros(Y.shape), []
    for f in range(SC.NF):
        inner = [j for j in range(SC.NF) if j != f]
        sse = np.zeros(len(ALPHAS))
        for j in inner:
            tr = np.isin(SC.FOLD, [i for i in inner if i != j])
            va = SC.FOLD == j
            Xv, Yv = X[va].reshape(-1, X.shape[-1]), Y[va].reshape(-1, q)
            for a_i, p in enumerate(_ridge_fit(X[tr].reshape(-1, X.shape[-1]), Y[tr].reshape(-1, q), ALPHAS)):
                sse[a_i] += ((p(Xv) - Yv) ** 2).sum()
        a = ALPHAS[int(np.argmin(sse))]
        chosen.append(a)
        tr, va = SC.FOLD != f, SC.FOLD == f
        out[va] = _ridge_fit(X[tr].reshape(-1, X.shape[-1]), Y[tr].reshape(-1, q), [a])[0](
            X[va].reshape(-1, X.shape[-1])).reshape(-1, T, q)
    return out, chosen


def hgb_cf(X, Y):
    N, _, q = Y.shape
    out = np.zeros(Y.shape)
    for f in range(SC.NF):
        tr, va = SC.FOLD != f, SC.FOLD == f
        Xt, Xv = X[tr].reshape(-1, X.shape[-1]), X[va].reshape(-1, X.shape[-1])
        for d in range(q):
            m = HistGradientBoostingRegressor(**HGB).fit(Xt, Y[tr][..., d].reshape(-1))
            out[va, :, d] = m.predict(Xv).reshape(-1, T)
    return out


def fit_all(Q, X, P2, P1, onebit, sets=SETS):
    """Q (N, T, 8) float32 -> {set: Q_hat (N, T, 8) float32}, {set: info}."""
    CH = R16.CH.astype(np.float64)
    Y = (Q.astype(np.float64) @ CH)[..., None] if onebit else Q.astype(np.float64)
    fits, info = {}, {}
    for s in sets:
        if s == 'F0':
            qh, _ = R16.pos_fit(Q, P2, P1)
            qh = qh.astype(np.float64)
            info[s] = {}
        else:
            if s in ('F1', 'F2', 'F3'):
                yh, al = ridge_cf(X[s], Y)
                info[s] = {'alpha': al}
            else:
                yh = hgb_cf(X[s], Y)
                info[s] = {}
            qh = yh * CH if onebit else yh
        info[s]['r2'] = r2(Q, qh)
        fits[s] = qh.astype(np.float32)
    return fits, info
