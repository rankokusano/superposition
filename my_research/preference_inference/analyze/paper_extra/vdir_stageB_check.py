"""
Paper round 13, V-dir stage B: checks before making the data (docs/paper_extra_log.md Sec.37.1; procedure a73e74a).
CPU only, stage-A saved arrays and dataset metadata; no model is run.
  1/2: N80 per pair (same formula as stage A Sec.6.3) and between-pair SD vs within-pair SE for b_pref (P1), its
       differences 50% - none (P2) and 50% - sign-randomised (P3; 50% c+- not saved -> proxies), and b_dir (D1-D3),
       in the stage-B main regression (stage-A model 'cov': initial positions on the axis), c+- FP pref 1-50 tau 1.0.
  4:   per (liked x layout x A-2 initial-y stratum) counts of direction A / B / none (tau 1.0, frames 1-50).
-> data/result/paper_extra/Vdir_B/check/summary.json
"""

import json
import os

import h5py
import numpy as np
from scipy import stats

import vdir_stageA as V
import vdir_stageA_add as VA

OUT = os.path.join(V.RES, 'paper_extra', 'Vdir_B', 'check')
W, TAU, ST = '1-50', 1.0, 'pref'
EDGE = 10.0 / 3.0
STRATA = (('B', -np.inf, -EDGE), ('mid', -EDGE, EDGE), ('A', EDGE, np.inf))


def pair_coefs(series, cond, err):
    pts = {}
    m = VA.M[ST]
    X, names, keep = V.design(m, W, TAU, 'cov', False)
    for s in V.PAIRS[cond]:
        z = V.load_pair(series, cond, err, ST, s)
        if z is None:
            continue
        qs = []
        for p in z:
            if series == 'oracle':
                LA = np.where(m['s_red'][:, None] == 1, p['Lr'], p['Lg'])
                LB = np.where(m['s_red'][:, None] == 1, p['Lg'], p['Lr'])
                qs.append(V.q_window(LB, W) - V.q_window(LA, W))
            else:
                qs.append(V.q_window(p['B'], W) - V.q_window(p['A'], W))
        pts[s] = np.array(qs)[:, keep]                     # (5, n)
    return X, names, pts


def boot(X, Ys):
    """Ys: list of (5, n) for the models compared within one pair; same episode resample for all.
    -> point coefficients (k,) per model and bootstrap (B, k) per model."""
    n = X.shape[0]
    idx = np.random.default_rng(0).integers(0, n, (V.B, n))
    Xb = X[idx]
    pt, bt = [], []
    for Y in Ys:
        pt.append(V.solve(X[None], Y.T[None])[0].mean(1))
        Yb = np.swapaxes(Y[:, idx], 0, 1)                 # (B, 5, n)
        bt.append(V.solve(Xb, np.swapaxes(Yb, 1, 2)).mean(2))
    return pt, bt


def term(names, nm):
    return names.index(nm)


def guide(vals, ses, n_used, n_pairs_ref=6):
    vals = np.array(vals)
    b, sd = float(vals.mean()), float(vals.std(ddof=1))
    se = float(np.mean(ses))
    n1200 = 1200 * n_used / 300
    se1200 = se * np.sqrt(n_used / n1200)
    nc = np.sqrt(n_pairs_ref) * abs(b) / sd
    tcrit = stats.t.ppf(0.975, n_pairs_ref - 1)
    power = float(1 - stats.nct.cdf(tcrit, n_pairs_ref - 1, nc) + stats.nct.cdf(-tcrit, n_pairs_ref - 1, nc))
    t = V.tsum(vals.tolist())
    return {'mean': b, 'ci95': t['ci95'], 'n_pairs': len(vals), 'per_pair': vals.tolist(), 'sd_pairs': sd,
            'se_within': se, 'n_used': n_used, 'N80': float(n_used * (2.8 * se / abs(b)) ** 2),
            'P80': float((2.8 * sd / abs(b)) ** 2), 'se_within_1200_proj': float(se1200),
            'n_used_1200_proj': float(n1200), 't_power_6pairs_obsSD': power}


def check12():
    R = {}
    specs = {  # name -> (series, err, cond_a, cond_b or None)
        'P1_D1': ('cpm', 'FP', 'p50', None),
        'P2_D2': ('cpm', 'FP', 'p50', 'none'),
        'P3_D3_proxy_a_cpm_p50-sr80': ('cpm', 'FP', 'p50', 'sr80'),
        'P3_D3_proxy_b_oracle_p50-sr50': ('oracle', 'FP', 'p50', 'sr50'),
        'P1_D1_oracle_p50': ('oracle', 'FP', 'p50', None),
    }
    for nm, (series, err, ca, cb) in specs.items():
        X, names, A = pair_coefs(series, err=err, cond=ca)
        Bm = pair_coefs(series, err=err, cond=cb)[2] if cb else None
        pairs = [s for s in A if Bm is None or s in Bm]
        out = {'series': series, 'err': err, 'cond_a': ca, 'cond_b': cb, 'pairs': [s + 1 for s in pairs]}
        for tm in ('pref', 'dir'):
            j = term(names, tm)
            vals, ses = [], []
            for s in pairs:
                Ys = [A[s]] + ([Bm[s]] if Bm is not None else [])
                pt, bt = boot(X, Ys)
                if Bm is None:
                    vals.append(pt[0][j]); ses.append(np.std(bt[0][:, j], ddof=1))
                else:
                    vals.append(pt[0][j] - pt[1][j]); ses.append(np.std(bt[0][:, j] - bt[1][:, j], ddof=1))
            out[tm] = guide(vals, ses, X.shape[0])
        R[nm] = out
        print(nm, json.dumps({k: {q: out[k][q] for q in ('mean', 'sd_pairs', 'se_within', 'N80', 'P80',
                                                         't_power_6pairs_obsSD')} for k in ('pref', 'dir')}), flush=True)
    return R


def counts(op, liked, layout):
    y0 = op[:, 0] @ V.NHAT
    p = (op[:, 50] - op[:, 0]) @ V.NHAT
    keep = (np.abs(p) >= TAU) & (p != 0)
    out = {}
    for lk in (1, 0):
        for L in (0, 1):
            for sn, lo, hi in STRATA:
                sel = (liked == lk) & (layout == L) & (y0 >= lo) & (y0 < hi)
                n = int(sel.sum())
                c = {'n': n, 'A': int((sel & keep & (p > 0)).sum()), 'B': int((sel & keep & (p < 0)).sum()),
                     'none': int((sel & ~keep).sum())}
                c['per100'] = {k: (100 * c[k] / n if n else None) for k in ('A', 'B', 'none')}
                out[f"{'red' if lk else 'green'}|L{L}|{sn}"] = c
    return out


def check4():
    R = {}
    for split in ('test', 'train'):
        g = h5py.File(os.path.join(V._PI, 'data', 'data', 'v11_s2_pref', 'data.h5'), 'r')[split]
        op = g['other_position'][:, :51].astype(np.float64)
        c = counts(op, g['a2_liked'][()], g['layout'][()])
        low = {k: v for k, v in c.items() if v['n'] == 0 or min(v['per100']['A'], v['per100']['B']) < 20}
        R[split] = {'cells': c, 'cells_below_20_per100': sorted(low), 'n_episodes': int(op.shape[0])}
        print(split, 'below 20:', sorted(low), flush=True)
    return R


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {'strata_edges': [-EDGE, EDGE], 'check12': check12(), 'check4': check4()}
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
