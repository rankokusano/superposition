"""
Paper round 13 rev2, Sec.1.6 (docs/paper_extra_log.md Sec.37.3 B; procedure afba126). CPU only.
  - checks on the new sign-randomised 50% c+- arrays (Vdir_B/c_sr50/inverse): ep200 repeat bit-identical; ref_red /
    ref_green equal to E2's sr50 true / opp rearranged by liked colour.
  - P3 / D3: N80 per pair etc. for b_pref / b_dir of (50% - sign-randomised 50%), c+- FP pref 1-50 tau 1.0, model
    'cov' (vdir_stageB_check.check12 machinery).
  - projections to the stage-B design (4 cells layout x liked, 300 each) from the stage-A set (test; train as
    reference): disagreeing episodes (s_dir != s_pref, direction present) overall and per A-2 initial-y third,
    no-direction count, weighted VIF of the main regression.
-> data/result/paper_extra/Vdir_B/check/summary2.json
"""

import json
import os

import h5py
import numpy as np

import vdir_stageA as V
import vdir_stageB_check as C

CSR = os.path.join(V.RES, 'paper_extra', 'Vdir_B', 'c_sr50', 'inverse')
V.CPM['FP']['sr50'] = (CSR, 'csr50mse_signrand50')
V.CPM['L1']['sr50'] = (CSR, 'csr50_signrand50')


def checks():
    R = {}
    a = np.load(os.path.join(CSR, 'csr50mse_signrand50_0_ep200_pref.npz'))
    b = np.load(os.path.join(CSR, 'csr50mse_signrand50_0_ep200_rep_pref.npz'))
    R['rep_bit_identical'] = all(np.array_equal(a[k], b[k]) for k in a.files)
    same = tot = 0
    for err, pre in (('FP', 'csr50mse_signrand50'), ('L1', 'csr50_signrand50')):
        for s in range(6):
            for e in V.EPS:
                z = np.load(os.path.join(CSR, f'{pre}_{s}_ep{e}_pref.npz'))
                w = np.load(os.path.join(V.RUNS, f'sr50_{V.ERRS[err]}_s{s}_ep{e}_pref.npz'))
                liked = w['a2_liked'][:, None] == 1
                Lr = np.where(liked, w[f'{err}_true'], w[f'{err}_opp'])
                Lg = np.where(liked, w[f'{err}_opp'], w[f'{err}_true'])
                tot += 1
                same += int(np.array_equal(Lr, z[f'{err}_ref_red']) and np.array_equal(Lg, z[f'{err}_ref_green']))
    R['ref_eq_E2_sr50'] = f'{same}/{tot}'
    m = C.VA.M['pref']
    R['correct_sign_eq_s_pref'] = bool(np.array_equal(a['correct_sign'], m['s_pref']))
    return R


def p3():
    R = {}
    for nm, (series, ca, cb) in {'P3_D3': ('cpm', 'p50', 'sr50'), 'sr50_alone_P_D': ('cpm', 'sr50', None)}.items():
        X, names, A = C.pair_coefs(series, ca, 'FP')
        Bm = C.pair_coefs(series, cb, 'FP')[2] if cb else None
        pairs = [s for s in A if Bm is None or s in Bm]
        out = {'pairs': [s + 1 for s in pairs], 'cond_a': ca, 'cond_b': cb}
        for tm in ('pref', 'dir'):
            j = names.index(tm)
            vals, ses = [], []
            for s in pairs:
                pt, bt = C.boot(X, [A[s]] + ([Bm[s]] if Bm is not None else []))
                if Bm is None:
                    vals.append(pt[0][j]); ses.append(np.std(bt[0][:, j], ddof=1))
                else:
                    vals.append(pt[0][j] - pt[1][j]); ses.append(np.std(bt[0][:, j] - bt[1][:, j], ddof=1))
            out[tm] = C.guide(vals, ses, X.shape[0])
        R[nm] = out
        print(nm, json.dumps({k: {q: out[k][q] for q in ('mean', 'ci95', 'sd_pairs', 'se_within', 'N80', 'P80',
                                                         't_power_6pairs_obsSD')} for k in ('pref', 'dir')}), flush=True)
    return R


def wvif(Z, w):
    """weighted VIF of the columns of Z (no intercept column) with weights w."""
    out = []
    sw = np.sqrt(w)
    for j in range(Z.shape[1]):
        o = np.column_stack([np.ones(len(Z))] + [Z[:, i] for i in range(Z.shape[1]) if i != j])
        y = Z[:, j]
        c, *_ = np.linalg.lstsq(o * sw[:, None], y * sw, rcond=None)
        r = y - o @ c
        mu = (w * y).sum() / w.sum()
        r2 = 1 - (w * r ** 2).sum() / (w * (y - mu) ** 2).sum()
        out.append(float(1 / (1 - r2)))
    return out


def project(split):
    g = h5py.File(os.path.join(V._PI, 'data', 'data', 'v11_s2_pref', 'data.h5'), 'r')[split]
    op = g['other_position'][:, :51].astype(np.float64)
    sp = g['self_position'][:, :1].astype(np.float64)
    liked, lay = g['a2_liked'][()], g['layout'][()]
    red_dir = {L: float(np.sign(V.LANDMARKS[V.CODE_LAYOUT[L]]['Red'] @ V.NHAT)) for L in (0, 1)}
    s_red = np.array([red_dir[int(L)] for L in lay])
    s_pref = np.where(liked == 1, s_red, -s_red)
    y0, y1 = op[:, 0] @ V.NHAT, sp[:, 0] @ V.NHAT
    p = (op[:, 50] - op[:, 0]) @ V.NHAT
    keep = (np.abs(p) >= C.TAU) & (p != 0)
    sd = np.sign(p)
    w = np.zeros(len(lay))
    cells = {}
    for lk in (0, 1):
        for L in (0, 1):
            sel = (liked == lk) & (lay == L)
            w[sel] = 300.0 / sel.sum()
            cells[f"{'red' if lk else 'green'}|L{L}"] = int(sel.sum())
    dis = keep & (sd != s_pref)
    R = {'n_source': int(len(lay)), 'cells_source': cells,
         'disagree_1200': float(w[dis].sum()), 'no_dir_1200': float(w[~keep].sum()),
         'dir_present_1200': float(w[keep].sum()), 'disagree_by_third': {}}
    for sn, lo, hi in C.STRATA:
        t = (y0 >= lo) & (y0 < hi)
        R['disagree_by_third'][sn] = {'disagree_1200': float(w[dis & t].sum()), 'all_1200': float(w[t].sum()),
                                      'dir_present_1200': float(w[keep & t].sum()), 'n_source_disagree': int((dis & t).sum())}
    Z = np.column_stack([s_pref, sd, s_red, y0, y1])[keep]
    R['vif_weighted'] = dict(zip(['pref', 'dir', 'red', 'a2y0', 'a1y0'], wvif(Z, w[keep])))
    R['vif_unweighted'] = dict(zip(['pref', 'dir', 'red', 'a2y0', 'a1y0'], wvif(Z, np.ones(keep.sum()))))
    print(split, json.dumps(R), flush=True)
    return R


def main():
    R = {'checks': checks()}
    print('checks', R['checks'], flush=True)
    R['p3'] = p3()
    R['projection'] = {sp: project(sp) for sp in ('test', 'train')}
    json.dump(R, open(os.path.join(C.OUT, 'summary2.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
