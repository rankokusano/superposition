"""
Paper round 13 rev2, V-dir stage B (docs/paper_extra_log.md Sec.37.5; pre-registration 4216497, data bd7cf3f).
Aggregation of the stage-B evaluation (Vdir_B/eval/inverse/<tag>_vdirb.npz, eval_inverse_v14 via vdirb_eval.py).
CPU only. Written and committed before any evaluation output was read.
  Q_e(W) = mean_W L(B pleasant) - mean_W L(A pleasant): c+- series (main; A = c+, B = c-) and oracle series (ref_red /
  ref_green, A pleasant = red-liked P if red is on side A). FP (mse stage, main) and L1 (l1 stage).
  Main regression (Sec.37.5.4): Q = a + b_pref s_pref + b_dir s_dir + b_red s_red + c2 y0_A2 + c1 y0_A1, no-direction
  (|p| < tau) dropped; pair value = mean over the 5 points of the per-point OLS coefficient.
  Judgements (Sec.37.5.5), c+- FP frames 1-50 tau 1.0: P1 b_pref(p50) > 0, P2 b_pref(p50 - none) > 0,
  P3 b_pref(p50 - sr50) > 0; D1-D3 for b_dir. Holds: 6-pair t CI lower > 0 AND two-stage bootstrap CI lower > 0;
  "partial" if only one; family claim only if all three hold.
  Two-stage bootstrap (1000, default_rng(0)): pairs resampled (6 of 6), per selected pair 1200 episodes resampled
  (same episodes for the two models of a difference and for the 5 points), no-direction episodes dropped inside the
  resample, refit, mean over the 6 selected pairs.
  Sensitivity (not judged): (i) + initial x of A-2 / A-1; (ii) + distances of A-2 / A-1 initial positions to the
  corners of A-2's liked / disliked colour; (iii) no initial position; (iv) A-2 initial y replaced by thirds (B / A
  dummies, middle reference, edges +-10/3) and thirds x s_dir; (v) s_dir replaced by the fraction of frames 1-50 with
  A-2 on side A (y > 0), same episodes as the main form.
  Secondary (not judged): (i) no-direction form on all 1200 episodes; (ii) two-choice BA (choose c+ if Q > 0, c- if
  Q < 0, 0.5 if equal; correct if = s_pref; mean over the 4 layout x liked cells; per point -> 5-point mean);
  (iii) the same choice's accuracy on the disagreeing episodes (direction present, s_dir != s_pref), overall and per
  A-2 initial-y third, with counts. Direction-label BA: the same choice against s_dir, direction present, mean over
  the 4 layout x liked cells.
  Per-pair point counts: per pair and point, within-pair episode bootstrap (1000, default_rng(0)); points whose CI
  lower > 0; pairs with >= 3 of 5.
  Bootstrap (two-stage, point counts) at frames 1-50, tau 1.0: all forms for c+- FP, the main form for the other
  series / errors; elsewhere the 6-pair t CI only.
-> data/result/paper_extra/Vdir_B/analysis/summary.json, tables.md
"""

import json
import os

import numpy as np

import vdir_stageA as V
import eval_s2_v11

eval_s2_v11.SETS['vdirb'] = ('paper_vdir_b_pref', 'test')
EV = os.path.join(V.RES, 'paper_extra', 'Vdir_B', 'eval', 'inverse')
OUT = os.path.join(V.RES, 'paper_extra', 'Vdir_B', 'analysis')
CONDS = ('none', 'p50', 'p80', 'sr80', 'sr50', 'sw')
PAIRS = {c: list(range(3)) if c == 'sw' else list(range(6)) for c in CONDS}
TAGS = {'none': ('v12mse_random', 'v12_random'), 'p50': ('v12mse_mixed', 'v12_mixed'), 'p80': ('v13mse_mixed', 'v13_mixed'),
        'sr80': ('csrmse_signrand', 'csr_signrand'), 'sr50': ('csr50mse_signrand50', 'csr50_signrand50'),
        'sw': ('v14cmse_mixedpref', 'v14c_mixedpref')}
ERRS = ('FP', 'L1')
SERIES = ('cpm', 'oracle')
PRIMARY = ('cpm', 'FP', '1-50', 1.0)
DIFFS = (('p50', 'none'), ('p50', 'sr50'), ('p80', 'none'), ('p80', 'sr80'))
JUDGE = {'P1': ('pref', 'p50', None), 'P2': ('pref', 'p50', 'none'), 'P3': ('pref', 'p50', 'sr50'),
         'D1': ('dir', 'p50', None), 'D2': ('dir', 'p50', 'none'), 'D3': ('dir', 'p50', 'sr50')}
FORMS = ('main', 's_i', 's_ii', 's_iii', 's_iv', 's_v', 'nodir')
EDGE = 10.0 / 3.0
B = 1000
CH = 50          # bootstrap chunk


# ------------------------------------------------------------------ data
def meta():
    m = V.meta('vdirb')
    import h5py
    g = h5py.File(V.load_set('vdirb')['path'], 'r')['test']
    op, sp = g['other_position'][()].astype(np.float64), g['self_position'][()].astype(np.float64)
    m['a2x0'], m['a1x0'] = op[:, 0, 0], sp[:, 0, 0]
    liked_c = np.where(m['liked'] == 1, 'Red', 'Green')
    dis_c = np.where(m['liked'] == 1, 'Green', 'Red')
    cl = np.array([V.LANDMARKS[V.CODE_LAYOUT[int(L)]][c] for L, c in zip(m['layout'], liked_c)])
    cd = np.array([V.LANDMARKS[V.CODE_LAYOUT[int(L)]][c] for L, c in zip(m['layout'], dis_c)])
    m['d2l'], m['d2d'] = np.linalg.norm(op[:, 0] - cl, axis=1), np.linalg.norm(op[:, 0] - cd, axis=1)
    m['d1l'], m['d1d'] = np.linalg.norm(sp[:, 0] - cl, axis=1), np.linalg.norm(sp[:, 0] - cd, axis=1)
    y2 = m['a2y0']
    m['third'] = np.where(y2 < -EDGE, 'B', np.where(y2 >= EDGE, 'A', 'mid'))
    m['fracA'] = ((op[:, 1:51] @ V.NHAT) > 0).mean(1)              # frames 1-50 on side A (y > 0)
    m['cell'] = m['liked'].astype(int) * 2 + m['layout'].astype(int)
    return m


def load_q(series, err, cond, m):
    """-> {window: (npairs, 5, 1200)} or None if a file is missing."""
    pre = TAGS[cond][0 if err == 'FP' else 1]
    out = {w: [] for w in V.WINS}
    for s in PAIRS[cond]:
        qs = {w: [] for w in V.WINS}
        for e in V.EPS:
            f = os.path.join(EV, f'{pre}_{s}_ep{e}_vdirb.npz')
            if not os.path.exists(f):
                return None
            z = np.load(f)
            assert np.array_equal(z['correct_sign'], m['s_pref']) and np.array_equal(z['layout'], m['layout'])
            if series == 'cpm':
                LA, LB = z[f'{err}_cp'].astype(np.float64), z[f'{err}_cm'].astype(np.float64)
            else:
                Lr, Lg = z[f'{err}_ref_red'].astype(np.float64), z[f'{err}_ref_green'].astype(np.float64)
                LA = np.where(m['s_red'][:, None] == 1, Lr, Lg)
                LB = np.where(m['s_red'][:, None] == 1, Lg, Lr)
            for w in V.WINS:
                qs[w].append(V.q_window(LB, w) - V.q_window(LA, w))
        for w in V.WINS:
            out[w].append(qs[w])
    return {w: np.array(v) for w, v in out.items()}


# ------------------------------------------------------------------ designs
def design(m, form, w, tau):
    p = m['p'][w]
    sd = np.sign(p)
    keep = (np.abs(p) >= tau) & (p != 0)
    c = {}
    if form == 'nodir':
        keep = np.ones_like(keep)
        c.update(pref=m['s_pref'], red=m['s_red'], a2y0=m['a2y0'], a1y0=m['a1y0'])
    elif form == 's_v':
        c.update(pref=m['s_pref'], dir=m['fracA'], red=m['s_red'], a2y0=m['a2y0'], a1y0=m['a1y0'])
    elif form == 's_iii':
        c.update(pref=m['s_pref'], dir=sd, red=m['s_red'])
    elif form == 's_iv':
        dB, dA = (m['third'] == 'B').astype(float), (m['third'] == 'A').astype(float)
        c.update(pref=m['s_pref'], dir=sd, red=m['s_red'], thirdB=dB, thirdA=dA, thirdB_x_dir=dB * sd,
                 thirdA_x_dir=dA * sd, a1y0=m['a1y0'])
    else:
        c.update(pref=m['s_pref'], dir=sd, red=m['s_red'], a2y0=m['a2y0'], a1y0=m['a1y0'])
        if form == 's_i':
            c.update(a2x0=m['a2x0'], a1x0=m['a1x0'])
        if form == 's_ii':
            c.update(d2_liked=m['d2l'], d2_disliked=m['d2d'], d1_liked=m['d1l'], d1_disliked=m['d1d'])
    names = ['a'] + list(c)
    X = np.column_stack([np.ones(len(p))] + [c[k] for k in names[1:]])
    return X, names, keep


def wsolve(X, Y, W):
    """X (..., n, k), Y (..., n, r), W (..., n) -> (..., k, r); nan where singular."""
    XtW = np.swapaxes(X * W[..., None], -1, -2)
    A, b = XtW @ X, XtW @ Y
    ok = np.linalg.cond(A) < 1e10
    A = np.where(ok[..., None, None], A, np.eye(A.shape[-1]))
    return np.where(ok[..., None, None], np.linalg.solve(A, b), np.nan)


def pair_coefs(X, keep, Q):
    """Q (np, 5, n) -> (np, k) 5-point means."""
    return wsolve(X[None], np.swapaxes(Q, 1, 2), keep[None].astype(float)).mean(2)


def pct(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else None


def tsum(v):
    return V.tsum(list(map(float, v)))


def two_stage(X, keep, Qa, Qb=None, npairs=6):
    """(B, k) bootstrap means of the pair coefficients (or of the within-pair differences a - b)."""
    rng = np.random.default_rng(0)
    pi = rng.integers(0, npairs, (B, npairs))
    ei = rng.integers(0, X.shape[0], (B, npairs, X.shape[0]))
    out = []
    for c0 in range(0, B, CH):
        p, e = pi[c0:c0 + CH], ei[c0:c0 + CH]
        Xb, Wb = X[e], keep[e].astype(float)                                    # (ch, np, n, k), (ch, np, n)
        res = []
        for Q in ((Qa,) if Qb is None else (Qa, Qb)):
            Yb = Q[p[:, :, None, None], np.arange(5)[None, None, :, None], e[:, :, None, :]]   # (ch, np, 5, n)
            res.append(wsolve(Xb, np.swapaxes(Yb, -1, -2), Wb).mean(-1))       # (ch, np, k)
        d = res[0] if Qb is None else res[0] - res[1]
        out.append(d.mean(1))
    return np.concatenate(out)


def point_counts(X, keep, Qa, Qb=None):
    """per pair: number of points whose within-pair episode-bootstrap CI lower > 0 (per coefficient)."""
    n = X.shape[0]
    idx = np.random.default_rng(0).integers(0, n, (B, n))
    Xb, Wb = X[idx], keep[idx].astype(float)
    cnt = []
    for i in range(Qa.shape[0]):
        r = []
        for Q in ((Qa,) if Qb is None else (Qa, Qb)):
            Yb = np.swapaxes(Q[i][:, idx], 0, 1)                                 # (B, 5, n)
            r.append(wsolve(Xb, np.swapaxes(Yb, 1, 2), Wb))                     # (B, k, 5)
        d = r[0] if Qb is None else r[0] - r[1]
        lo = np.nanpercentile(d, 2.5, axis=0)                                    # (k, 5)
        cnt.append((lo > 0).sum(1).tolist())
    return cnt


def coef_block(X, names, keep, Qa, Qb=None, boot=True, npairs=6):
    pv = pair_coefs(X, keep, Qa) - (pair_coefs(X, keep, Qb) if Qb is not None else 0)
    blk = {'names': names, 'n_used': int(keep.sum()), 'per_pair': {}, 't': {}}
    for j, nm in enumerate(names):
        blk['per_pair'][nm] = pv[:, j].tolist()
        blk['t'][nm] = tsum(pv[:, j])
    if 'pref' in names and 'dir' in names:
        d = pv[:, names.index('dir')] - pv[:, names.index('pref')]
        blk['per_pair']['dir_minus_pref'], blk['t']['dir_minus_pref'] = d.tolist(), tsum(d)
    if boot:
        bt = two_stage(X, keep, Qa, Qb, npairs)
        blk['boot2'] = {nm: pct(bt[:, j]) for j, nm in enumerate(names)}
        if 'pref' in names and 'dir' in names:
            blk['boot2']['dir_minus_pref'] = pct(bt[:, names.index('dir')] - bt[:, names.index('pref')])
        cn = point_counts(X, keep, Qa, Qb)
        blk['points_lower_gt0'] = {nm: [c[j] for c in cn] for j, nm in enumerate(names)}
        blk['pairs_ge3of5'] = {nm: int(sum(c[j] >= 3 for c in cn)) for j, nm in enumerate(names)}
    return blk


# ------------------------------------------------------------------ BA / disagreeing episodes
def choice_acc(Q, label, mask, cells):
    """Q (..., n): accuracy of sign(Q) against label (0.5 for ties), mean over the 4 cells, within mask."""
    hit = np.where(Q == 0, 0.5, (np.sign(Q) == label).astype(float))
    accs = [hit[..., mask & (cells == c)].mean(-1) for c in range(4) if (mask & (cells == c)).any()]
    return np.mean(accs, 0)


def ba_values(Q, m, keep):
    """Q (np, 5, n) -> dict of (np,) pair values (5-point means)."""
    dis = keep & (np.sign(m['p']['1-50']) != m['s_pref'])
    allm = np.ones_like(keep)
    out = {'BA_pref': choice_acc(Q, m['s_pref'], allm, m['cell']).mean(1),
           'BA_dir': choice_acc(Q, np.sign(m['p']['1-50']), keep, m['cell']).mean(1)}
    hit = np.where(Q == 0, 0.5, (np.sign(Q) == m['s_pref']).astype(float))
    out['acc_disagree'] = hit[..., dis].mean(-1).mean(1)
    for t in ('B', 'mid', 'A'):
        mk = dis & (m['third'] == t)
        out[f'acc_disagree_{t}'] = hit[..., mk].mean(-1).mean(1)
    return out


def ba_two_stage(Qa, m, keep, Qb=None, npairs=6):
    rng = np.random.default_rng(0)
    pi = rng.integers(0, npairs, (B, npairs))
    ei = rng.integers(0, len(keep), (B, npairs, len(keep)))
    res = {k: [] for k in ('BA_pref', 'BA_dir', 'acc_disagree', 'acc_disagree_B', 'acc_disagree_mid', 'acc_disagree_A')}
    lab_p, lab_d = m['s_pref'], np.sign(m['p']['1-50'])
    for b in range(B):
        vals = []
        for Q in ((Qa,) if Qb is None else (Qa, Qb)):
            v = {k: [] for k in res}
            for j in range(npairs):
                e = ei[b, j]
                q = Q[pi[b, j]][:, e]                                            # (5, n)
                kp, cl, th = keep[e], m['cell'][e], m['third'][e]
                lp, ld = lab_p[e], lab_d[e]
                v['BA_pref'].append(choice_acc(q, lp, np.ones_like(kp), cl).mean())
                v['BA_dir'].append(choice_acc(q, ld, kp, cl).mean())
                hit = np.where(q == 0, 0.5, (np.sign(q) == lp).astype(float))
                dis = kp & (ld != lp)
                v['acc_disagree'].append(hit[:, dis].mean() if dis.any() else np.nan)
                for t in ('B', 'mid', 'A'):
                    mk = dis & (th == t)
                    v[f'acc_disagree_{t}'].append(hit[:, mk].mean() if mk.any() else np.nan)
            vals.append({k: np.nanmean(x) for k, x in v.items()})
        for k in res:
            res[k].append(vals[0][k] - (vals[1][k] if Qb is not None else 0))
    return {k: pct(np.array(v)) for k, v in res.items()}


# ------------------------------------------------------------------ main
def judge(t, b2):
    a = t['ci95'] is not None and t['ci95'][0] > 0
    b = b2 is not None and b2[0] > 0
    return 'holds' if a and b else ('partial' if a or b else 'does not hold')


def main():
    os.makedirs(OUT, exist_ok=True)
    m = meta()
    R = {'checks': {}, 'missing': [], 'coef': {}, 'diff': {}, 'judgement': {}, 'secondary': {}, 'counts': {}}
    # rep check
    a = np.load(os.path.join(EV, 'v12mse_mixed_0_ep200_vdirb.npz'))
    b = np.load(os.path.join(EV, 'v12mse_mixed_0_ep200_rep_vdirb.npz'))
    R['checks']['rep_bit_identical'] = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
    R['checks']['rep_all'] = all(R['checks']['rep_bit_identical'].values())
    keep1 = (np.abs(m['p']['1-50']) >= 1.0) & (m['p']['1-50'] != 0)
    R['counts'] = {'n_dir': int(keep1.sum()), 'n_disagree': int((keep1 & (np.sign(m['p']['1-50']) != m['s_pref'])).sum()),
                   'by_third': {t: int((keep1 & (np.sign(m['p']['1-50']) != m['s_pref']) & (m['third'] == t)).sum())
                                for t in ('B', 'mid', 'A')}}
    for series in SERIES:
        for err in ERRS:
            Q = {}
            for c in CONDS:
                q = load_q(series, err, c, m)
                if q is None:
                    R['missing'].append(f'{series}|{err}|{c}')
                else:
                    Q[c] = q
            for w in V.WINS:
                for tau in V.TAUS:
                    prim = (series, err, w, tau) == PRIMARY
                    rich = w == '1-50' and tau == 1.0
                    for form in (FORMS if rich else ('main',)):
                        X, names, keep = design(m, form, w, tau)
                        bt_on = rich and (form == 'main' or (series, err) == ('cpm', 'FP'))
                        for c in Q:
                            key = f'{series}|{err}|{w}|{tau}|{form}|{c}'
                            R['coef'][key] = coef_block(X, names, keep, Q[c][w], boot=bt_on, npairs=len(PAIRS[c]))
                        for ca, cb in DIFFS:
                            if ca in Q and cb in Q:
                                key = f'{series}|{err}|{w}|{tau}|{form}|{ca}-{cb}'
                                R['diff'][key] = coef_block(X, names, keep, Q[ca][w], Q[cb][w], boot=bt_on)
                    if rich:
                        for c in Q:
                            v = ba_values(Q[c][w], m, keep1)
                            blk = {k: {'per_pair': x.tolist(), 't': tsum(x)} for k, x in v.items()}
                            if series == 'cpm' and err == 'FP':
                                bt = ba_two_stage(Q[c][w], m, keep1, npairs=len(PAIRS[c]))
                                for k in blk:
                                    blk[k]['boot2'] = bt[k]
                            R['secondary'][f'{series}|{err}|{c}'] = blk
                        for ca, cb in DIFFS:
                            if ca in Q and cb in Q:
                                va, vb = ba_values(Q[ca][w], m, keep1), ba_values(Q[cb][w], m, keep1)
                                blk = {k: {'per_pair': (va[k] - vb[k]).tolist(), 't': tsum(va[k] - vb[k])} for k in va}
                                if series == 'cpm' and err == 'FP':
                                    bt = ba_two_stage(Q[ca][w], m, keep1, Q[cb][w])
                                    for k in blk:
                                        blk[k]['boot2'] = bt[k]
                                R['secondary'][f'{series}|{err}|{ca}-{cb}'] = blk
                    if prim:
                        for jn, (term, ca, cb) in JUDGE.items():
                            blk = R['coef' if cb is None else 'diff'][f'cpm|FP|1-50|1.0|main|{ca}' + ('' if cb is None else f'-{cb}')]
                            R['judgement'][jn] = {'term': term, 'cond': ca, 'vs': cb, 'mean': blk['t'][term]['mean'],
                                                  't_ci95': blk['t'][term]['ci95'], 'boot2_ci95': blk['boot2'][term],
                                                  'per_pair': blk['per_pair'][term],
                                                  'pairs_ge3of5': blk['pairs_ge3of5'][term],
                                                  'verdict': judge(blk['t'][term], blk['boot2'][term])}
                        for fam in ('P', 'D'):
                            R['judgement'][f'family_{fam}'] = all(R['judgement'][f'{fam}{i}']['verdict'] == 'holds' for i in (1, 2, 3))
            print('done', series, err, flush=True)
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), ensure_ascii=False, indent=0)
    print('judgement', json.dumps(R['judgement'], ensure_ascii=False))
    print('checks rep_all', R['checks']['rep_all'], 'missing', R['missing'])


if __name__ == '__main__':
    main()
