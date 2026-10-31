"""
Paper round 14 (stage C): analyses AFTER the stage-C results (docs/paper_extra_log.md Sec.38.12; Kusano-san's request
of 2026-11-01). All of them are OUTSIDE the pre-registration (3ade5ed + 1de4a37). CPU only, original episodes; the
pre-registered aggregation (stageC_analysis.py, summary.json) is not changed or rerun.
  A  H0 (h2 with Q2 = 0 at every step):
     A1 code check: H0 must not depend on G (identical for every checkpoint of a pair and for the pre-training G built on
        the same stage I; it may differ between stage-I conditions). Bit identity of the saved H0 arrays.
     A2 the C (alpha) chosen by the nested CV for every read-out (basic, adjusted, direction, fraction, generalisation),
        from summary.json, and the same table for every representation.
     A3 out-of-fold predicted probabilities on the original episodes with the chosen C (as stageC_analysis.readout):
        basic (1200), adjusted (1088), direction (1088), agree -> disagreeing (280); quantiles, SD, share in
        [0.45, 0.55]; for H0 and, for comparison, H; the covariate-only baseline. The pair values of BA / dLL /
        gen_dis / dirBA are recomputed and compared with summary.json (reproduction).
     A4 initial position: H0 -> y0 of A-2 / A-1 (ridge, nested alpha, 5 folds, R^2); direction and preference read from
        the initial positions alone (y0 A2, y0 A1, x0 A2, x0 A1; logistic, nested C, 5 folds; BA).
  B  one-bit u_t (pre-registered existing quantity, Sec.38.5.5, missing from the aggregation code; computed after
     seeing the other results): episode choice c+ if sum_{t=1..50} u_t > 0, c- if < 0, 0.5 if 0 (analyze/v14/
     u_direction_v14.py); share following preference among the 280 disagreeing episodes (stage-B labels: s_pref,
     s_dir tau 1.0 frames 1-50); also agreement with preference (1200) and with direction (1088). Pair value = mean
     over ep160-200; t 95% CI over the 3 pairs (df 2); two-stage bootstrap 95% (1000, default_rng(0): 3 pairs with
     replacement, one resample of the episodes of the subset shared by the selected pairs and checkpoints).
-> data/result/paper_extra/StageC/posthoc/summary.json
"""

import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa

import stageC_analysis as S  # noqa  (labels, folds, read-out helpers; nothing is recomputed there)

OUT = os.path.join(S.A.V.RES, 'paper_extra', 'StageC', 'posthoc')
SUMM = json.load(open(os.path.join(S.OUT, 'summary.json')))
QS = (0, 5, 25, 50, 75, 95, 100)


def dist(p):
    p = np.asarray(p, float)
    return {'q': {str(q): float(np.percentile(p, q)) for q in QS}, 'sd': float(p.std()),
            'share_045_055': float(((p >= 0.45) & (p <= 0.55)).mean()), 'share_gt_05': float((p > 0.5).mean()),
            'n': int(p.size)}


def oof(X, C):
    """out-of-fold probabilities on the original episodes, as stageC_analysis.readout (idx = all)."""
    L = S.LAB
    y, yd, kp, fo = S.YP, S.YD, L['keep'], S.FOLD
    XC = np.hstack([S._cov(), X])
    pq, pa, pdr = np.full(S.N, np.nan), np.full(S.N, np.nan), np.full(S.N, np.nan)
    for f in range(S.NF):
        tr, va = fo != f, fo == f
        pq[va] = S._fit(X[tr], y[tr], X[va], C['basic'][f])
        tr2, va2 = tr & kp, va & kp
        pa[va2] = S._fit(XC[tr2], y[tr2], XC[va2], C['full'][f])
        pdr[va2] = S._fit(X[tr2], yd[tr2], X[va2], C['dir'][f])
    agree, dis = kp & (L['dir'] == L['pref']), L['dis']
    pg = S._fit(X[agree], y[agree], X[dis], C['gen_dis'])
    return pq, pa, pdr, pg


def part_a(R):
    D = S.load_all()
    _, Rep, _, _ = S.arrays(D)
    out = {}
    # A1
    a1 = {}
    for j in range(4):
        h = Rep['p50']['H0'][j]
        a1[f'pair{j + 1}'] = {
            'p50_ckpts_identical': bool(all(np.array_equal(h[0], h[k]) for k in range(1, h.shape[0]))),
            'p50_equals_pre': bool(np.array_equal(h[0], Rep['pre']['H0'][j][0])),
            'none_ckpts_identical': bool(all(np.array_equal(Rep['none']['H0'][j][0], Rep['none']['H0'][j][k])
                                             for k in range(1, 5))),
            'none_equals_pre_none': bool(np.array_equal(Rep['none']['H0'][j][0], Rep['pre_none']['H0'][j][0])),
            'p50_equals_none': bool(np.array_equal(h[0], Rep['none']['H0'][j][0])),
            'H0_sd_over_episodes_mean': float(h[0].std(0).mean()),
            'H_sd_over_episodes_mean': float(Rep['p50']['H'][j][0].std(0).mean())}
    out['A1'] = a1
    # A2: chosen C per representation / key / read-out kind
    a2 = {}
    for k, v in SUMM['C_chosen'].items():
        rep, key = k.split('|')[:2]
        for kind, c in v.items():
            for x in (c if isinstance(c, list) else [c]):
                kk = f'{rep}|{key}|{kind}'
                a2.setdefault(kk, {})
                a2[kk][str(x)] = a2[kk].get(str(x), 0) + 1
    out['A2'] = {'per_rep_key_kind': a2, 'baseline_C': SUMM['C_base']}
    # A3
    a3 = {}
    yp = S.YP
    for rep in ('H0', 'H'):
        for key in ('p50', 'none', 'pre'):
            arr = Rep[key][rep]
            P = {'basic': [], 'adjusted': [], 'direction': [], 'gen_dis': []}
            pair_vals = {'BA': [], 'dLL': [], 'gen_dis': [], 'pos_dirBA': []}
            llb = S.baseline_ll(np.arange(S.N), SUMM['C_base'])
            for j in range(arr.shape[0]):
                pts = {m: [] for m in pair_vals}
                for k in range(arr.shape[1]):
                    C = SUMM['C_chosen'][f'{rep}|{key}|{j}|{k}']
                    pq, pa, pdr, pg = oof(arr[j, k], C)
                    kp = S.LAB['keep']
                    P['basic'].append(pq); P['adjusted'].append(pa[kp]); P['direction'].append(pdr[kp])
                    P['gen_dis'].append(pg)
                    pts['BA'].append(S._ba(pq, yp, S.LAB['cell']))
                    pts['dLL'].append(llb - S._ll(pa[kp], yp[kp]))
                    pts['gen_dis'].append(float(((pg > .5) == yp[S.LAB['dis']]).mean()))
                    dcell = (S.YD * 2 + S.LAB['lay']).astype(int)
                    pts['pos_dirBA'].append(S._ba(pdr[kp], S.YD[kp], dcell[kp]))
                for m in pair_vals:
                    pair_vals[m].append(float(np.mean(pts[m])))
            saved = {m: SUMM['readout'][f'nest|{rep}|{key}|{m}']['t']['per_pair'] for m in pair_vals}
            a3[f'{rep}|{key}'] = {
                'prob': {kind: dist(np.concatenate(v)) for kind, v in P.items()},
                'pair_values_recomputed': pair_vals,
                'max_abs_diff_vs_summary': {m: float(np.max(np.abs(np.array(pair_vals[m]) - np.array(saved[m]))))
                                            for m in pair_vals}}
            print('A3', rep, key, {m: round(v, 6) for m, v in a3[f'{rep}|{key}']['max_abs_diff_vs_summary'].items()},
                  flush=True)
    # baseline (covariates only) probabilities
    kp = S.LAB['keep']
    pb = np.full(S.N, np.nan)
    for f in range(S.NF):
        tr, va = (S.FOLD != f) & kp, (S.FOLD == f) & kp
        pb[va] = S._fit(S._cov()[tr], yp[tr], S._cov()[va], SUMM['C_base'][f])
    a3['baseline_covariates'] = {'prob': dist(pb[kp])}
    out['A3'] = a3
    # A4: initial position
    L = S.LAB
    a4 = {}
    Xpos = np.column_stack([L['y2'], L['y1'], L['x2'], L['x1']])
    for nm, X in (('initial_positions', Xpos),):
        res = {}
        for lab, y, sel in (('direction', S.YD, kp), ('preference', yp, np.ones(S.N, bool))):
            pr = np.full(S.N, np.nan)
            Cs = []
            for f in range(S.NF):
                tr = np.where((S.FOLD != f) & sel)[0]
                va = (S.FOLD == f) & sel
                c = S.select(X, y, tr, 'logit')
                Cs.append(c)
                pr[va] = S._fit(X[tr], y[tr], X[va], c)
            cell = (y * 2 + L['lay']).astype(int) if lab == 'direction' else L['cell']
            res[lab] = {'BA': S._ba(pr[sel], y[sel], cell[sel]), 'C': Cs}
        agree, dis = kp & (L['dir'] == L['pref']), L['dis']
        c = S.select(X, yp, np.where(agree)[0], 'logit')
        pg = S._fit(X[agree], yp[agree], X[dis], c)
        res['gen_dis'] = {'acc': float(((pg > .5) == yp[dis]).mean()), 'C': c}
        a4[nm] = res
    h0r = {}
    for key in ('p50', 'none'):
        for tgt in ('y2', 'y1', 'x2', 'x1'):
            vals = []
            for j in range(4):
                X = Rep[key]['H0'][j][0]
                y = L[tgt]
                pr = np.zeros(S.N)
                for f in range(S.NF):
                    tr = np.where(S.FOLD != f)[0]
                    va = S.FOLD == f
                    pr[va] = S._ridge(X[tr], y[tr], X[va], S.select(X, y, tr, 'ridge'))
                vals.append(float(1 - ((y - pr) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
            h0r[f'{key}|{tgt}'] = vals
    a4['H0_to_initial_position_R2_by_pair'] = h0r
    out['A4'] = a4
    print('A4', json.dumps(a4)[:600], flush=True)
    R['A'] = out


def tci(v):
    return S.tci(v, '95')


def part_b(R):
    L = S.LAB
    dis, kp = L['dis'], L['keep']
    out = {}
    rng = np.random.default_rng(0)
    for key in ('onebit50', 'onebit80'):
        exp, eps, np_ = S.SEC[key]
        CH = np.array([[np.load(os.path.join(S.EV, f'{exp}_s{s}_ep{e}.npz'))['u'][:, 1:51].astype(np.float64).sum(1)
                        for e in eps] for s in range(np_)])                      # (P, K, N)
        chp = np.where(CH > 0, 1.0, np.where(CH < 0, 0.0, 0.5))
        agp = np.where(L['pref'] > 0, chp, 1 - chp)
        agd = np.where(L['dir'] > 0, chp, 1 - chp)
        res = {'n_ties': int((CH == 0).sum())}
        for nm, a, sel in (('follow_pref_disagreeing', agp, dis), ('agree_pref_all', agp, np.ones(S.N, bool)),
                           ('agree_dir_1088', agd, kp)):
            per = a[:, :, sel].mean(2).mean(1)
            idx = np.where(sel)[0]
            B = []
            for _ in range(1000):
                pi = rng.integers(0, np_, np_)
                ei = idx[rng.integers(0, len(idx), len(idx))]
                B.append(float(np.mean([a[j][:, ei].mean() for j in pi])))
            res[nm] = {'t': tci(per), 'boot': S.bci(np.array(B), '95'),
                       'per_ckpt': a[:, :, sel].mean(2).tolist()}
        out[key] = res
        print('B', key, {k: (v['t']['mean'], v['t']['ci'], v['boot']) for k, v in res.items() if k != 'n_ties'}, flush=True)
    R['B'] = out


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {'note': 'outside the pre-registration (Sec.38.12)'}
    part_b(R)
    part_a(R)
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
