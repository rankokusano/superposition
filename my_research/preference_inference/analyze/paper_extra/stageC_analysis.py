"""
Paper round 14 (stage C, G diagnosis): aggregation of StageC/eval/*.npz (stageC_eval.py) per the pre-registration
(docs/paper_extra_log.md Sec.38.5, commit 3ade5ed). CPU only (nice, <= 16 processes).
  labels / covariates: as stage B (vdirb_analysis.meta: s_pref, s_red, s_dir (tau 1.0, frames 1-50), fraction of
  frames 1-50 on side A, initial y / x of A-2 and A-1, distances to the liked / disliked corners, initial-y thirds).
  (L1) q_e = mean over steps 1-50 of Q2_t . c+/|c+|; q~_e standardised per model x checkpoint on the episodes used
       (1200 for stage 1, 1088 for stage 2), recomputed in every bootstrap replicate; sd < 1e-6 -> missing (pair value =
       mean of the non-missing points, 0 if all missing).
       Stage 1: q~ = a + b_pref s_pref + b_red s_red + c2 y0_A2 + c1 y0_A1 (1200). Stage 2: + b_dir s_dir (1088).
       Q1a b_pref(50%) > 0; Q1b 50% - none > 0; Q1c 50% trained - pre-training (00000.pth) > 0; Q2a / Q2b at stage 2.
       Holds: t CI lower > 0 AND two-stage bootstrap CI lower > 0; levels (Sec.38.5.4): Q1a/Q1b 95% (t 0.975, df 3;
       2.5/97.5), Q1c and Q2a/Q2b 97.5% (t 0.9875; 1.25/98.75); stage 1 = Q1a AND Q1b; Q1c and stage 2 judged only if
       stage 1 holds.
  (L2)(L3) read-outs of Q (8), S (128), O (64), H (128), H0 (128) (per-episode means over steps 1-50): basic (BA over
       the 4 layout x preference cells, log loss; 1200) and adjusted (Delta LL = LL(baseline: s_dir, fraction on A,
       y0 A2, y0 A1, layout) - LL(baseline + representation); 1088); generalisation (agree -> 280 disagreeing; layout
       0 -> 1, 1 -> 0); positive controls (direction BA, 1088; fraction-on-A R^2, L2 ridge). 5 folds stratified
       by layout x preference (default_rng(0), once), per-fold standardisation on the training side.
       Regularisation (amendment Sec.38.9): L2 logistic C chosen by nested 5-fold CV (inner folds over original
       episode indices, stratified, rng(0)) from {1e-4, 1e-3, 1e-2, 1e-1, 1} by inner log loss, ridge alpha from
       {1e-2, 1e-1, 1, 10, 100} by inner squared error; chosen ONCE on the original data per outer fold / fit and
       fixed in the bootstrap (the choice's variability is not in the CI). C = alpha = 1.0 as sensitivity ('c1').
       Refitted (standardisation and coefficients) in every bootstrap replicate for L1, L2 AND L3 (Sec.38.5.1).
  Two-stage bootstrap (1000, default_rng(0)): 4 pairs with replacement; ONE resample of the 1200 original episode
       indices per replicate shared by every selected pair, condition and checkpoint (pre-training included);
       duplicates stay in their original fold.
  Sanity check (before any real-data read-out; Sec.38.9): artificial 8 / 64 / 128-dim features (label + noise; noise
       only) through the same read-out code, on the original episodes and on one resample; pass if (a) BA >= 0.70 and
       Delta LL > 0.05, (b) 0.45 <= BA <= 0.55 and |Delta LL| <= 0.02 (criteria of 588591e).
Modes: --sanity | --run [--reps 1000 --procs 16]
-> data/result/paper_extra/StageC/analysis/{sanity_nested.json, summary.json, boot_draws.npz}
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np  # noqa
from scipy import stats  # noqa
from sklearn.linear_model import LogisticRegression, Ridge  # noqa

import vdirb_analysis as A  # noqa  (stage-B labels; registers 'vdirb')

EV = os.path.join(A.V.RES, 'paper_extra', 'StageC', 'eval')
OUT = os.path.join(A.V.RES, 'paper_extra', 'StageC', 'analysis')
CP = np.array(json.load(open(os.path.join(A.V.RES, 'v14', 'cand_shape.json')))['c_plus'], np.float64)
CPU_ = CP / np.linalg.norm(CP)
EPS = (160, 170, 180, 190, 200)
N, NF = 1200, 5
MAIN = {'p50': ('v12_s2_pref_mixed', EPS), 'none': ('v12_s2_pref_random', EPS), 'pre': ('v12_s2_pref_mixed', (0,)),
        'pre_none': ('v12_s2_pref_random', (0,))}
SEC = {'p80free': ('v13_s2_pref_mixed', EPS, 4), 'onebit50': ('v14_s2u_pref_mixed', EPS, 3),
       'onebit80': ('v14_s2u_pref_mixed80', EPS, 3), 'green50': ('v12_s2_green_mixed', EPS, 4),
       'green_none': ('v12_s2_green_random', EPS, 4)}
REPS = ('Q', 'S', 'O', 'H', 'H0')
L1FORMS = ('s1', 's2', 's1_1088', 'v', 'i', 'ii', 'iii', 'iv')
LEVEL = {'95': (0.975, 2.5, 97.5), '97.5': (0.9875, 1.25, 98.75)}

# ------------------------------------------------------------------ labels
M = A.meta()
P_DIR = M['p']['1-50']
LAB = {'pref': M['s_pref'].astype(float), 'red': M['s_red'].astype(float), 'dir': np.sign(P_DIR),
       'keep': (np.abs(P_DIR) >= 1.0) & (P_DIR != 0), 'frac': M['fracA'], 'y2': M['a2y0'], 'y1': M['a1y0'],
       'x2': M['a2x0'], 'x1': M['a1x0'], 'd2l': M['d2l'], 'd2d': M['d2d'], 'd1l': M['d1l'], 'd1d': M['d1d'],
       'tB': (M['third'] == 'B').astype(float), 'tA': (M['third'] == 'A').astype(float),
       'lay': M['layout'].astype(float), 'cell': M['cell']}
LAB['dis'] = LAB['keep'] & (LAB['dir'] != LAB['pref'])
FOLD = np.empty(N, int)
_r = np.random.default_rng(0)
for _c in range(4):
    _i = np.where(LAB['cell'] == _c)[0]
    _r.shuffle(_i)
    FOLD[_i] = np.arange(len(_i)) % NF


# ------------------------------------------------------------------ data
def load_all():
    D = {}
    for key, (exp, eps) in MAIN.items():
        D[key] = [[np.load(os.path.join(EV, f'{exp}_s{s}_ep{e}.npz')) for e in eps] for s in range(4)]
    for key, (exp, eps, np_) in SEC.items():
        D[key] = [[np.load(os.path.join(EV, f'{exp}_s{s}_ep{e}.npz')) for e in eps] for s in range(np_)]
    return D


def arrays(D):
    Q, R, G, NRM = {}, {}, {}, {}
    for key, pairs in D.items():
        Q[key] = np.array([[(z['Q2'][:, 1:51].astype(np.float64) @ CPU_).mean(1) for z in pts] for pts in pairs])  # (P,K,N)
        NRM[key] = np.array([[np.linalg.norm(z['Q2'][:, 1:51], axis=-1).mean() for z in pts] for pts in pairs])
        G[key] = np.array([[z['ge_tot_G_1_50'].astype(np.float64) for z in pts] for pts in pairs])
        if key in MAIN:
            R[key] = {r: np.array([[z[f'{r}_m1_50'].astype(np.float64) for z in pts] for pts in pairs]) for r in REPS}
    return Q, R, G, NRM


# ------------------------------------------------------------------ L1
def design(form, idx):
    L = {k: v[idx] for k, v in LAB.items()}
    one = np.ones(len(idx))
    base = [one, L['pref'], L['red'], L['y2'], L['y1']]
    kp = L['keep']
    if form == 's1':
        return np.column_stack(base), np.ones(len(idx), bool)
    if form == 's1_1088':
        return np.column_stack(base), kp
    if form == 's2':
        return np.column_stack(base + [L['dir']]), kp
    if form == 'v':
        return np.column_stack(base + [L['frac']]), kp
    if form == 'i':
        return np.column_stack(base + [L['dir'], L['x2'], L['x1']]), kp
    if form == 'ii':
        return np.column_stack(base + [L['dir'], L['d2l'], L['d2d'], L['d1l'], L['d1d']]), kp
    if form == 'iii':
        return np.column_stack([one, L['pref'], L['dir'], L['red']]), kp
    if form == 'iv':
        return np.column_stack([one, L['pref'], L['dir'], L['red'], L['tB'], L['tA'], L['tB'] * L['dir'],
                                L['tA'] * L['dir'], L['y1']]), kp
    raise ValueError(form)


def l1_pair(qk, idx, form, standardize=True):
    """qk (K, N) -> pair value of b_pref (and b_dir if present): mean over checkpoints, missing rule."""
    X, sel = design(form, idx)
    Y = qk[:, idx][:, sel].T                                       # (n, K)
    X = X[sel]
    if standardize:
        sd = Y.std(0)
        ok = sd >= 1e-6
        Y = (Y - Y.mean(0)) / np.where(ok, sd, 1.0)
    else:
        ok = np.ones(Y.shape[1], bool)
    c = np.linalg.lstsq(X, Y, rcond=None)[0]                       # (k, K)
    out = {}
    for j, nm in ((1, 'pref'), (2, 'dir')):
        if nm == 'dir' and form in ('s1', 's1_1088'):
            continue
        v = c[j][ok]
        out[nm] = float(v.mean()) if len(v) else 0.0
    out['n_missing'] = int((~ok).sum())
    return out


def l1_all(Q, idx, pairs_sel=None, keys=('p50', 'none', 'pre', 'pre_none')):
    """-> {(key, form, std): array over selected pairs of the b_pref / b_dir pair values}"""
    out = {}
    for key in keys:
        P = Q[key].shape[0]
        ps = range(P) if pairs_sel is None else pairs_sel
        for form in L1FORMS:
            for std in (True, False):
                vals = [l1_pair(Q[key][j], idx, form, std) for j in ps]
                for nm in ('pref', 'dir'):
                    if nm in vals[0]:
                        out[(key, form, std, nm)] = np.array([v[nm] for v in vals])
                if std:
                    out[(key, form, std, 'n_missing')] = np.array([v['n_missing'] for v in vals])
    return out


# ------------------------------------------------------------------ read-outs (Sec.38.9: nested-CV C / alpha)
CGRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
AGRID = (1e-2, 1e-1, 1.0, 10.0, 100.0)


def _std(Xtr, Xva):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (Xtr - mu) / sd, (Xva - mu) / sd


def _fit(Xtr, ytr, Xva, C=1.0):
    a, b = _std(Xtr, Xva)
    m = LogisticRegression(C=C, max_iter=2000).fit(a, ytr)
    return m.predict_proba(b)[:, 1]


def _ridge(Xtr, ytr, Xva, alpha=1.0):
    a, b = _std(Xtr, Xva)
    return Ridge(alpha=alpha).fit(a, ytr).predict(b)


def _ba(prob, y, cell):
    hit = (prob > 0.5) == (y > 0)
    return float(np.mean([hit[cell == c].mean() for c in range(4) if (cell == c).any()]))


def _ll(prob, y):
    p = np.clip(prob, 1e-12, 1 - 1e-12)
    return float(-np.mean(np.where(y > 0, np.log(p), np.log(1 - p))))


def inner_folds(orig_idx):
    """5 inner folds over the given ORIGINAL episode indices (unique), stratified by layout x preference, rng(0)."""
    orig_idx = np.unique(orig_idx)
    f = np.full(N, -1)
    r = np.random.default_rng(0)
    for c in range(4):
        i = orig_idx[LAB['cell'][orig_idx] == c]
        r.shuffle(i)
        f[i] = np.arange(len(i)) % NF
    return f


def select(X, y, orig_train, kind):
    """choose C (logistic, log loss) or alpha (ridge, squared error) by inner 5-fold CV on the original episodes."""
    fi = inner_folds(orig_train)
    tr_all = np.unique(orig_train)
    best, bv = None, np.inf
    for v in (CGRID if kind == 'logit' else AGRID):
        pred = np.zeros(N)
        for g in range(NF):
            tr, va = tr_all[fi[tr_all] != g], tr_all[fi[tr_all] == g]
            pred[va] = _fit(X[tr], y[tr], X[va], v) if kind == 'logit' else _ridge(X[tr], y[tr], X[va], v)
        p = pred[tr_all]
        loss = _ll(p, y[tr_all]) if kind == 'logit' else float(((p - y[tr_all]) ** 2).mean())
        if loss < bv - 1e-12:
            best, bv = v, loss
    return best


def _cov():
    return np.column_stack([LAB['dir'], LAB['frac'], LAB['y2'], LAB['y1'], LAB['lay']])


YP = (LAB['pref'] > 0).astype(int)
YD = (LAB['dir'] > 0).astype(int)


def choose_base():
    kp = LAB['keep']
    return [select(_cov(), YP, np.where((FOLD != f) & kp)[0], 'logit') for f in range(NF)]


def choose(X):
    """per outer fold: C for basic / adjusted (cov + X) / direction, alpha for fraction; per generalisation fit."""
    kp = LAB['keep']
    XC = np.hstack([_cov(), X])
    C = {'basic': [], 'full': [], 'dir': [], 'ridge': []}
    for f in range(NF):
        tr = np.where(FOLD != f)[0]
        tr2 = np.where((FOLD != f) & kp)[0]
        C['basic'].append(select(X, YP, tr, 'logit'))
        C['full'].append(select(XC, YP, tr2, 'logit'))
        C['dir'].append(select(X, YD, tr2, 'logit'))
        C['ridge'].append(select(X, LAB['frac'], tr, 'ridge'))
    agree = np.where(kp & (LAB['dir'] == LAB['pref']))[0]
    C['gen_dis'] = select(X, YP, agree, 'logit')
    C['gen_lay01'] = select(X, YP, np.where(LAB['lay'] == 0)[0], 'logit')
    C['gen_lay10'] = select(X, YP, np.where(LAB['lay'] == 1)[0], 'logit')
    return C


C1 = {'basic': [1.0] * NF, 'full': [1.0] * NF, 'dir': [1.0] * NF, 'ridge': [1.0] * NF,
      'gen_dis': 1.0, 'gen_lay01': 1.0, 'gen_lay10': 1.0}
CB1 = [1.0] * NF


def baseline_ll(idx, CB):
    L = {k: v[idx] for k, v in LAB.items()}
    y = YP[idx]
    cov = _cov()[idx]
    kp, fo = L['keep'], FOLD[idx]
    pb = np.zeros(len(idx))
    for f in range(NF):
        tr, va = (fo != f) & kp, (fo == f) & kp
        pb[va] = _fit(cov[tr], y[tr], cov[va], CB[f])
    return _ll(pb[kp], y[kp])


def readout(X, idx, llb, C, full=True):
    """X (N, d) original-episode features -> metrics on the resample idx, with C / alpha fixed per outer fold."""
    L = {k: v[idx] for k, v in LAB.items()}
    X = X[idx]
    y, yd = YP[idx], YD[idx]
    cov = _cov()[idx]
    kp, fo, cell = L['keep'], FOLD[idx], L['cell']
    n = len(idx)
    pq, pa, pdr, pr = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    XC = np.hstack([cov, X])
    for f in range(NF):
        tr, va = fo != f, fo == f
        pq[va] = _fit(X[tr], y[tr], X[va], C['basic'][f])
        tr2, va2 = tr & kp, va & kp
        pa[va2] = _fit(XC[tr2], y[tr2], XC[va2], C['full'][f])
        if full:
            pdr[va2] = _fit(X[tr2], yd[tr2], X[va2], C['dir'][f])
            pr[va] = _ridge(X[tr], L['frac'][tr], X[va], C['ridge'][f])
    r = {'BA': _ba(pq, y, cell), 'LL': _ll(pq, y), 'dLL': llb - _ll(pa[kp], y[kp])}
    if full:
        agree, dis = kp & (L['dir'] == L['pref']), L['dis']
        r['gen_dis'] = float(((_fit(X[agree], y[agree], X[dis], C['gen_dis']) > .5) == y[dis]).mean())
        l0, l1 = L['lay'] == 0, L['lay'] == 1
        r['gen_lay01'] = float(((_fit(X[l0], y[l0], X[l1], C['gen_lay01']) > .5) == y[l1]).mean())
        r['gen_lay10'] = float(((_fit(X[l1], y[l1], X[l0], C['gen_lay10']) > .5) == y[l0]).mean())
        dcell = (yd * 2 + L['lay']).astype(int)
        r['pos_dirBA'] = _ba(pdr[kp], yd[kp], dcell[kp])
        fr = L['frac']
        r['pos_fracR2'] = float(1 - ((fr - pr) ** 2).sum() / ((fr - fr.mean()) ** 2).sum())
    return r


METRICS = ('BA', 'LL', 'dLL', 'gen_dis', 'gen_lay01', 'gen_lay10', 'pos_dirBA', 'pos_fracR2')
ROKEYS = ('p50', 'none', 'pre', 'pre_none')


def arr_list(R):
    """[(rep, key, pair, ckpt)] of the arrays to read out; O only once (frozen encoder: identical for every model)."""
    out = [('O', 'p50', 0, 0)]
    for rep in REPS:
        if rep == 'O':
            continue
        for key in ROKEYS:
            P, K = R[key][rep].shape[:2]
            out += [(rep, key, j, k) for j in range(P) for k in range(K)]
    return out


def readout_all(R, idx, CS, CB, variant, pairs_sel=None):
    """CS: {(rep, key, pair, ckpt): C dict}; variant 'nest' (chosen C) or 'c1' (C = alpha = 1)."""
    llb = baseline_ll(idx, CB if variant == 'nest' else CB1)
    rO = readout(R['p50']['O'][0, 0], idx, llb, CS[('O', 'p50', 0, 0)] if variant == 'nest' else C1)
    out = {}
    for rep in REPS:
        for key in ROKEYS:
            arr = R[key][rep]
            ps = range(arr.shape[0]) if pairs_sel is None else pairs_sel
            vals = []
            for j in ps:
                pts = [rO if rep == 'O' else
                       readout(arr[j, k], idx, llb, CS[(rep, key, j, k)] if variant == 'nest' else C1)
                       for k in range(arr.shape[1])]
                vals.append({m: float(np.mean([p[m] for p in pts])) for m in pts[0]})
            for m in vals[0]:
                out[(rep, key, m)] = np.array([v[m] for v in vals])
    return out


# ------------------------------------------------------------------ CI helpers
def tci(v, level):
    v = np.asarray(v, float)
    n = len(v)
    q = LEVEL[level][0]
    se = v.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(q, n - 1) * se
    return {'mean': float(v.mean()), 'ci': [float(v.mean() - h), float(v.mean() + h)], 'per_pair': v.tolist(), 'n': n}


def bci(d, level):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    lo, hi = LEVEL[level][1], LEVEL[level][2]
    return [float(np.percentile(d, lo)), float(np.percentile(d, hi))]


# ------------------------------------------------------------------ workers
_G = {}


def _init(Q, R, CS, CB):
    _G['Q'], _G['R'], _G['CS'], _G['CB'] = Q, R, CS, CB


def _choose_job(args):
    rep, key, j, k = args
    return args, choose(_G['R'][key][rep][j, k])


def _rep(args):
    pi, ei = args
    Q, R = _G['Q'], _G['R']
    r1 = l1_all(Q, ei, list(pi))
    res = {'l1': {}, 'ro': {}}
    for (key, form, std, nm), v in r1.items():
        if nm != 'n_missing':
            res['l1'][(key, form, std, nm)] = float(v.mean())
    for form in L1FORMS:
        for std in (True, False):
            for nm in ('pref', 'dir'):
                a = r1.get(('p50', form, std, nm))
                if a is None:
                    continue
                res['l1'][('p50-none', form, std, nm)] = float((a - r1[('none', form, std, nm)]).mean())
                res['l1'][('p50-pre', form, std, nm)] = float((a - r1[('pre', form, std, nm)]).mean())
    for variant in ('nest', 'c1'):
        ro = readout_all(R, ei, _G['CS'], _G['CB'], variant, list(pi))
        for (rep, key, m), v in ro.items():
            res['ro'][(variant, rep, key, m)] = float(v.mean())
        for rep in REPS:
            for m in METRICS:
                res['ro'][(variant, rep, 'p50-none', m)] = float((ro[(rep, 'p50', m)] - ro[(rep, 'none', m)]).mean())
                res['ro'][(variant, rep, 'p50-pre', m)] = float((ro[(rep, 'p50', m)] - ro[(rep, 'pre', m)]).mean())
    return res


# ------------------------------------------------------------------ sanity (Sec.38.9: 8 / 64 / 128 dims, nested C)
def sanity():
    y = LAB['pref']
    idx = np.arange(N)
    ei = np.random.default_rng(0).integers(0, N, N)
    CB = choose_base()
    out = {'criteria': 'a: BA >= 0.70 and dLL > 0.05; b: 0.45 <= BA <= 0.55 and |dLL| <= 0.02 (588591e)',
           'base_C': CB}
    allok = True
    for d in (8, 64, 128):
        rng = np.random.default_rng(123)
        Xa = rng.normal(size=(N, d)) + 0.5 * y[:, None] * rng.normal(size=d)[None]
        Xb = rng.normal(size=(N, d))
        for nm, X in (('a_label_plus_noise', Xa), ('b_noise_only', Xb)):
            Cx = choose(X)
            res = {'C_chosen': Cx}
            for where, ix in (('original', idx), ('resample', ei)):
                for variant, Cv, Cb in (('nest', Cx, CB), ('c1', C1, CB1)):
                    llb = baseline_ll(ix, Cb)
                    res[f'{where}_{variant}'] = readout(X, ix, llb, Cv)
            for where in ('original', 'resample'):
                r = res[f'{where}_nest']
                ok = (r['BA'] >= 0.70 and r['dLL'] > 0.05) if nm.startswith('a') else \
                     (0.45 <= r['BA'] <= 0.55 and abs(r['dLL']) <= 0.02)
                res[f'pass_{where}'] = bool(ok)
                allok &= ok
            out[f'{d}|{nm}'] = res
            print(d, nm, {w: (round(res[f'{w}_nest']['BA'], 3), round(res[f'{w}_nest']['dLL'], 4), res[f'pass_{w}'])
                          for w in ('original', 'resample')}, 'C', Cx['basic'], Cx['full'], flush=True)
    out['pass_all'] = bool(allok)
    for d in (8, 64, 128):
        out[f'pass_{d}'] = all(out[f'{d}|{nm}'][f'pass_{w}'] for nm in ('a_label_plus_noise', 'b_noise_only')
                               for w in ('original', 'resample'))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'sanity_nested.json'), 'w'), indent=1)
    print({k: v for k, v in out.items() if k.startswith('pass')})


# ------------------------------------------------------------------ run
def run(reps, procs):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    D = load_all()
    Q, R, G, NRM = arrays(D)
    idx = np.arange(N)
    S = {'counts': {'n': N, 'n_dir': int(LAB['keep'].sum()), 'n_dis': int(LAB['dis'].sum())}}
    CB = choose_base()
    _init(Q, R, {}, CB)
    with Pool(procs, initializer=_init, initargs=(Q, R, {}, CB)) as pool:
        CS = dict(pool.map(_choose_job, arr_list(R), chunksize=4))
    S['C_base'] = CB
    S['C_chosen'] = {'|'.join(map(str, k)): v for k, v in CS.items()}
    S['t_choose_s'] = time.time() - t0
    p1 = l1_all(Q, idx)
    pro = {v: readout_all(R, idx, CS, CB, v) for v in ('nest', 'c1')}
    rng = np.random.default_rng(0)
    draws = [(rng.integers(0, 4, 4), rng.integers(0, N, N)) for _ in range(reps)]
    with Pool(procs, initializer=_init, initargs=(Q, R, CS, CB)) as pool:
        B = pool.map(_rep, draws, chunksize=max(1, reps // (procs * 4)))
    S['t_boot_s'] = time.time() - t0

    def bdraws(sec, key):
        return np.array([b[sec][key] for b in B])
    L1 = {}
    for form in L1FORMS:
        for std in (True, False):
            for nm in ('pref', 'dir'):
                if ('p50', form, std, nm) not in p1:
                    continue
                for key in ('p50', 'none', 'pre', 'pre_none', 'p50-none', 'p50-pre'):
                    if key == 'p50-none':
                        v = p1[('p50', form, std, nm)] - p1[('none', form, std, nm)]
                    elif key == 'p50-pre':
                        v = p1[('p50', form, std, nm)] - p1[('pre', form, std, nm)]
                    else:
                        v = p1[(key, form, std, nm)]
                    bk = (key, form, std, nm)
                    bd = bdraws('l1', bk) if bk in B[0]['l1'] else None
                    L1[f'{key}|{form}|{"std" if std else "raw"}|{nm}'] = {
                        lv: {'t': tci(v, lv), 'boot': bci(bd, lv) if bd is not None else None} for lv in ('95', '97.5')}
    S['L1'] = L1
    S['n_missing'] = {f'{k}|{f}': p1[(k, f, True, 'n_missing')].tolist() for k in MAIN for f in L1FORMS}

    def verdict(key, lv):
        e = L1[key][lv]
        a, b = e['t']['ci'][0] > 0, e['boot'][0] > 0
        return 'holds' if a and b else ('partial' if a or b else 'does not hold')
    J = {'Q1a': verdict('p50|s1|std|pref', '95'), 'Q1b': verdict('p50-none|s1|std|pref', '95'),
         'Q1c': verdict('p50-pre|s1|std|pref', '97.5'), 'Q2a': verdict('p50|s2|std|pref', '97.5'),
         'Q2b': verdict('p50-none|s2|std|pref', '97.5')}
    J['stage1'] = J['Q1a'] == 'holds' and J['Q1b'] == 'holds'
    J['Q1c_and_stage2_judged'] = J['stage1']
    J['stage2'] = J['stage1'] and J['Q2a'] == 'holds' and J['Q2b'] == 'holds'
    S['judgement'] = J
    RO = {}
    for variant in ('nest', 'c1'):
        for rep in REPS:
            for key in ('p50', 'none', 'pre', 'pre_none', 'p50-none', 'p50-pre'):
                for m in METRICS:
                    pv = pro[variant]
                    if key == 'p50-none':
                        v = pv[(rep, 'p50', m)] - pv[(rep, 'none', m)]
                    elif key == 'p50-pre':
                        v = pv[(rep, 'p50', m)] - pv[(rep, 'pre', m)]
                    else:
                        v = pv[(rep, key, m)]
                    bk = (variant, rep, key, m)
                    bd = bdraws('ro', bk) if bk in B[0]['ro'] else None
                    RO[f'{variant}|{rep}|{key}|{m}'] = {'t': tci(v, '95'), 'boot': bci(bd, '95') if bd is not None else None}
    S['readout'] = RO
    desc = {}
    for key in list(MAIN) + list(SEC):
        q = Q[key]
        sg = np.sign(q) == LAB['pref']
        desc[key] = {'q_mean_by_pair': q.mean((1, 2)).tolist(), 'q_sd_by_pair': q.std(2).mean(1).tolist(),
                     'norm_Q2_by_pair': NRM[key].mean(1).tolist(),
                     'sign_agree_all': tci(sg.mean(2).mean(1), '95'),
                     'sign_agree_dis': tci(sg[:, :, LAB['dis']].mean(2).mean(1), '95')}
        if key in SEC or key == 'pre_none':
            for form in ('s1', 's2'):
                for std in (True, False):
                    vals = [l1_pair(q[j], idx, form, std)['pref'] for j in range(q.shape[0])]
                    desc[key][f'b_pref_{form}_{"std" if std else "raw"}'] = tci(vals, '95')
        g = G[key]
        for form in ('s1', 's2', 'v'):
            desc[key][f'grad_b_pref_{form}'] = tci([l1_pair(g[j], idx, form, False)['pref'] for j in range(g.shape[0])], '95')
    S['descriptive'] = desc
    rng2 = np.random.default_rng(0)
    PP = {k: [] for k in ('p50|s1', 'p50-none|s1', 'p50-pre|s1', 'p50|s2', 'p50-none|s2')}
    for _ in range(reps):
        pi = rng2.integers(0, 4, 4)
        acc = {k: [] for k in PP}
        for j in pi:
            e = rng2.integers(0, N, N)
            for form in ('s1', 's2'):
                b = {key: l1_pair(Q[key][j], e, form)['pref'] for key in ('p50', 'none', 'pre')}
                acc[f'p50|{form}'].append(b['p50'])
                acc[f'p50-none|{form}'].append(b['p50'] - b['none'])
                if form == 's1':
                    acc['p50-pre|s1'].append(b['p50'] - b['pre'])
        for k in PP:
            PP[k].append(np.mean(acc[k]))
    S['L1_per_pair_resample_sensitivity'] = {k: {'95': bci(v, '95'), '97.5': bci(v, '97.5')} for k, v in PP.items()}
    d = A.eval_s2_v11.load_set('vdirb')
    PA2 = A.eval_s2_v11.true_pleasure('vdirb', d)
    for key in list(MAIN) + list(SEC):
        S['descriptive'][key]['corr_Q2_PA2'] = tci([np.mean([A.eval_s2_v11.corr_q2(z['Q2'], PA2, d['layout'])['all']
                                                            for z in pts]) for pts in D[key]], '95')
    json.dump(S, open(os.path.join(OUT, 'summary.json'), 'w'), indent=0)
    np.savez_compressed(os.path.join(OUT, 'boot_draws.npz'),
                        l1_keys=np.array(['|'.join(map(str, k)) for k in B[0]['l1']]),
                        l1=np.array([[b['l1'][k] for k in B[0]['l1']] for b in B]),
                        ro_keys=np.array(['|'.join(map(str, k)) for k in B[0]['ro']]),
                        ro=np.array([[b['ro'][k] for k in B[0]['ro']] for b in B]))
    print('judgement', json.dumps(J), 't', time.time() - t0, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--reps', type=int, default=1000)
    ap.add_argument('--procs', type=int, default=16)
    a = ap.parse_args()
    if a.sanity:
        sanity()
    if a.run:
        run(a.reps, a.procs)


if __name__ == '__main__':
    main()
