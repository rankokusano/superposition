"""
Paper round 15 (docs/paper_extra_log.md Sec.39; ALL outside the pre-registration, descriptive only): on the stage-B
evaluation set (vdirb, 1200 episodes) used for stage C. CPU only (nice, <= 16 processes); no model is run.
  D1 behaviour-only baselines for the preference read-out, with the stage-C (L3) procedure unchanged
     (stageC_analysis: 5 outer folds stratified by layout x preference, per-fold standardisation, L2 logistic C chosen
     by nested 5-fold CV on the original episodes from {1e-4 .. 1} by inner log loss, fixed in the resamples):
       B1 summary cues: s_dir (sign of the frame 0 -> 50 displacement on the A axis, as stage C 'dir'), fraction of
          frames 1-50 on side A, A-2 initial position (x, y), layout                                     (5 dims)
       B2 trajectory:   A-2 position (x, y) at frames 0-50                                               (102 dims)
     metrics: basic BA over the 4 layout x preference cells (1200); agree -> disagree generalisation (fit on the
     episodes whose direction agrees with the preference, accuracy on the 280 disagreeing ones), as stage C.
     CI: episode resampling, 1000 replicates. The replicates use the stage-C draw sequence (default_rng(0); each
     replicate draws 4 pair indices, unused here, then 1200 episode indices), so replicate r has the same episodes as
     replicate r of StageC/analysis/boot_draws.npz; checked by recomputing the (O) read-out on replicates 0-4.
     Supplementary: paired differences read-out - baseline per replicate (read-out side: the stage-C two-stage draws).
  D2 per-episode Spearman correlations, pairs 1-4, per pair: stage-B Q_e (FP, c+-, steps 0-49, mean of the MSE-stage
     ep160-200) vs stage-C q_e (Q2 . c+/|c+| over steps 1-50, mean of G ep160-200) and vs the (L2) basic read-out
     probability of Q (out-of-fold, chosen C from StageC summary.json, mean over the G points); 50% G, no-experience G
     (with the no-experience stage-B Q_e) and pre-training G (ep0; with the 50% stage-B Q_e).
Modes: d1 | d2 | all   [--reps 1000 --procs 8]
-> data/result/paper_extra/Round15/{d1.json, d1_draws.npz, d2.json}
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

import stageC_analysis as SC  # noqa  (labels, folds, read-out helpers; imports vdirb_analysis)

A = SC.A
OUT = os.path.join(A.V.RES, 'paper_extra', 'Round15')
SCA = os.path.join(A.V.RES, 'paper_extra', 'StageC', 'analysis')
KEYS = ('p50', 'none', 'pre')
REPS = ('O', 'S', 'Q', 'H')


# ------------------------------------------------------------------ D1
def features():
    import h5py
    g = h5py.File(A.V.load_set('vdirb')['path'], 'r')['test']
    op = g['other_position'][()].astype(np.float64)
    assert op.shape[:2] == (SC.N, 101), op.shape
    assert np.allclose(SC.LAB['x2'], op[:, 0, 0]) and np.allclose(SC.LAB['y2'], op[:, 0, 1])
    b1 = np.column_stack([SC.LAB['dir'], SC.LAB['frac'], op[:, 0, 0], op[:, 0, 1], SC.LAB['lay']])
    b2 = op[:, 0:51].reshape(SC.N, -1)
    return {'B1': b1, 'B2': b2}


def choose(X):
    C = {'basic': [SC.select(X, SC.YP, np.where(SC.FOLD != f)[0], 'logit') for f in range(SC.NF)]}
    agree = np.where(SC.LAB['keep'] & (SC.LAB['dir'] == SC.LAB['pref']))[0]
    C['gen_dis'] = SC.select(X, SC.YP, agree, 'logit')
    return C


def metrics(X, idx, C):
    """BA and agree -> disagree accuracy on the resample idx; same lines as stageC_analysis.readout."""
    L = {k: v[idx] for k, v in SC.LAB.items()}
    Xi, y, fo = X[idx], SC.YP[idx], SC.FOLD[idx]
    pq = np.zeros(len(idx))
    for f in range(SC.NF):
        tr, va = fo != f, fo == f
        pq[va] = SC._fit(Xi[tr], y[tr], Xi[va], C['basic'][f])
    agree, dis = L['keep'] & (L['dir'] == L['pref']), L['dis']
    gen = float(((SC._fit(Xi[agree], y[agree], Xi[dis], C['gen_dis']) > .5) == y[dis]).mean())
    return {'BA': SC._ba(pq, y, L['cell']), 'gen_dis': gen}


_G = {}


def _init(F, CS):
    _G['F'], _G['CS'] = F, CS


def _rep(ei):
    return {b: metrics(_G['F'][b], ei, _G['CS'][b]) for b in _G['F']}


def stagec_draws(reps):
    rng = np.random.default_rng(0)
    return [(rng.integers(0, 4, 4), rng.integers(0, SC.N, SC.N))[1] for _ in range(reps)]


def pct(d):
    d = np.asarray(d, float)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def d1(reps, procs):
    t0 = time.time()
    F = features()
    CS = {b: choose(X) for b, X in F.items()}
    idx = np.arange(SC.N)
    point = {b: metrics(F[b], idx, CS[b]) for b in F}
    draws = stagec_draws(reps)
    # check: the (O) read-out recomputed on replicates 0-4 equals the saved stage-C draws
    S = json.load(open(os.path.join(SCA, 'summary.json')))
    z = np.load(os.path.join(SCA, 'boot_draws.npz'))
    rk = list(z['ro_keys'])
    XO = np.load(os.path.join(SC.EV, 'v12_s2_pref_mixed_s0_ep160.npz'))['O_m1_50'].astype(np.float64)
    CO = {'basic': S['C_chosen']['O|p50|0|0']['basic'], 'gen_dis': S['C_chosen']['O|p50|0|0']['gen_dis']}
    chk = []
    for r in range(5):
        v = metrics(XO, draws[r], CO)
        chk.append({m: [v[m], float(z['ro'][r, rk.index(f'nest|O|p50|{m}')])] for m in ('BA', 'gen_dis')})
    same = all(abs(a - b) < 1e-12 for c in chk for a, b in c.values())
    with Pool(procs, initializer=_init, initargs=(F, CS)) as pool:
        B = pool.map(_rep, draws, chunksize=max(1, reps // (procs * 4)))
    R = {'note': 'outside the pre-registration (Sec.39), descriptive', 'n': SC.N,
         'n_agree': int((SC.LAB['keep'] & (SC.LAB['dir'] == SC.LAB['pref'])).sum()), 'n_dis': int(SC.LAB['dis'].sum()),
         'dims': {b: int(X.shape[1]) for b, X in F.items()}, 'C_chosen': CS, 'draw_check': {'same': same, 'values': chk},
         'baseline': {}, 'readout': {}, 'diff': {}}
    for b in F:
        R['baseline'][b] = {m: {'point': point[b][m], 'ci': pct([x[b][m] for x in B])} for m in ('BA', 'gen_dis')}
    for rep in REPS:
        for key in KEYS:
            for m in ('BA', 'gen_dis'):
                ro = S['readout'][f'nest|{rep}|{key}|{m}']
                R['readout'][f'{rep}|{key}|{m}'] = {'point': ro['t']['mean'], 't': ro['t']['ci'], 'boot': ro['boot']}
                rd = z['ro'][:reps, rk.index(f'nest|{rep}|{key}|{m}')]
                for b in F:
                    bd = np.array([x[b][m] for x in B])
                    R['diff'][f'{rep}|{key}|{m}|{b}'] = {'point': ro['t']['mean'] - point[b][m], 'ci': pct(rd - bd)}
    R['t_s'] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, 'd1.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(OUT, 'd1_draws.npz'),
                        **{f'{b}|{m}': np.array([x[b][m] for x in B]) for b in F for m in ('BA', 'gen_dis')})
    print('d1', json.dumps(R['baseline']), 'draw check', same, 't', round(R['t_s'], 1), flush=True)


# ------------------------------------------------------------------ D2
def oof_prob(X, Cb):
    p = np.zeros(SC.N)
    for f in range(SC.NF):
        tr, va = SC.FOLD != f, SC.FOLD == f
        p[va] = SC._fit(X[tr], SC.YP[tr], X[va], Cb[f])
    return p


def d2():
    t0 = time.time()
    m = A.meta()
    QE = {c: A.load_q('cpm', 'FP', c, m)['1-50'][:4].mean(1) for c in ('p50', 'none')}   # (4, 1200)
    S = json.load(open(os.path.join(SCA, 'summary.json')))
    R = {'note': 'outside the pre-registration (Sec.39), descriptive; stage-B Q_e: stage-I MSE ep160-200 (no G); '
                 'stage-C q_e and read-out: G ep160-200 (pre: G ep0) on stage-I L1 ep200', 'per_pair': {}, 'summary': {}}
    for key in KEYS:
        exp, eps = SC.MAIN[key]
        qe_b = QE['none' if key == 'none' else 'p50']
        rows = []
        for s in range(4):
            qs, pr = [], []
            for k, e in enumerate(eps):
                z = np.load(os.path.join(SC.EV, f'{exp}_s{s}_ep{e}.npz'))
                qs.append((z['Q2'][:, 1:51].astype(np.float64) @ SC.CPU_).mean(1))
                pr.append(oof_prob(z['Q_m1_50'].astype(np.float64), S['C_chosen'][f'Q|{key}|{s}|{k}']['basic']))
            qs, pr = np.mean(qs, 0), np.mean(pr, 0)
            rows.append({'Qe_vs_qe': float(stats.spearmanr(qe_b[s], qs)[0]),
                         'Qe_vs_prob': float(stats.spearmanr(qe_b[s], pr)[0]),
                         'qe_vs_prob': float(stats.spearmanr(qs, pr)[0])})
        R['per_pair'][key] = rows
        R['summary'][key] = {q: {'mean': float(np.mean([r[q] for r in rows])),
                                 'range': [float(min(r[q] for r in rows)), float(max(r[q] for r in rows))]}
                             for q in rows[0]}
    R['t_s'] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, 'd2.json'), 'w'), indent=1)
    print('d2', json.dumps(R['summary']), 't', round(R['t_s'], 1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('d1', 'd2', 'all'))
    ap.add_argument('--reps', type=int, default=1000)
    ap.add_argument('--procs', type=int, default=8)
    a = ap.parse_args()
    assert a.procs <= 16
    if a.mode in ('d1', 'all'):
        d1(a.reps, a.procs)
    if a.mode in ('d2', 'all'):
        d2()


if __name__ == '__main__':
    main()
