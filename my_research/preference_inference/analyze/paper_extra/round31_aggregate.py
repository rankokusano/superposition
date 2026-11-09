"""
Paper round 31 aggregation (docs/paper_extra_log.md Sec.55; procedure fixed before computing).
Reads Round31/main/grid.npz, Round31/read/read.npz, Round31/check/check.json
-> Round31/analysis/summary.json, tables.md, fig_round31_examples.{png,pdf}, fig_round31_d.{png,pdf}

Units: grid placements without overlap (2 layouts x 361 x 360 = 259,920). The decoded images do not depend on the
predictor (Encoders identical in all 24 predictors, checked), so the 6-pair t CI has zero width; the main CI is the
two-stage bootstrap (pairs -> placements, 1000, seed 0), whose pair stage has no effect here. Descriptive: bootstrap
of clusters (layout x A-2 position, 722 clusters).
Correlations pool placements x ranges of the range set; zero variance -> undefined (counted as low).
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
OUT = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round31')
V12_JSON = os.path.join(_PI, 'data', 'result', 'v12', 'main', 'fig5', 'fig5_v12_fig5_s0_s0_ep100.json')
ORIG_EXP2 = {'self_self': 0.051295, 'other_other': 0.112475, 'self_other': 0.139270, 'other_self': 0.146516}  # result.txt
BINSETS = {'c0167': [0, 1, 6, 7], 'all': list(range(8)), **{f'b{b}': [b] for b in range(8)}}
MEAS = ('pos_thr', 'pos_cont', 'col_thr', 'col_cont')
NB = 1000


def stats(x, y, bins):
    """(N, 8) x2 -> per-placement (N, 7): n, Sx, Sy, Sxx, Syy, Sxy, SSE(y - x)."""
    x, y = x[:, bins].astype(np.float64), y[:, bins].astype(np.float64)
    return np.stack([np.full(x.shape[0], len(bins), np.float64), x.sum(1), y.sum(1), (x * x).sum(1), (y * y).sum(1),
                     (x * y).sum(1), ((y - x) ** 2).sum(1)], 1)


def r_r2(S):
    """S (..., 7) summed -> (r, R2) with NaN when undefined."""
    n, sx, sy, sxx, syy, sxy, sse = np.moveaxis(S, -1, 0)
    vx, vy = sxx - sx * sx / n, syy - sy * sy / n
    with np.errstate(invalid='ignore', divide='ignore'):
        r = np.where((vx > 1e-12 * n) & (vy > 1e-12 * n), (sxy - sx * sy / n) / np.sqrt(vx * vy), np.nan)
        r2 = np.where(vy > 1e-12 * n, 1 - sse / vy, np.nan)
    return r, r2


def boot_counts(N, B, rs, clusters=None, chunk=50):
    """yield (b, N) weight matrices: placements resampled with replacement, or whole clusters."""
    for c0 in range(0, B, chunk):
        nb = min(chunk, B - c0)
        if clusters is None:
            yield np.stack([np.bincount(rs.randint(0, N, N), minlength=N) for _ in range(nb)]).astype(np.float64)
        else:
            K = clusters.max() + 1
            yield np.stack([np.bincount(rs.randint(0, K, K), minlength=K)[clusters] for _ in range(nb)]).astype(np.float64)


def ci(a):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) > NB // 2 else [None, None]


def main():
    z = np.load(os.path.join(OUT, 'main', 'grid.npz'))
    rd = np.load(os.path.join(OUT, 'read', 'read.npz'))
    chk = json.load(open(os.path.join(OUT, 'check', 'check.json')))
    od = os.path.join(OUT, 'analysis')
    os.makedirs(od, exist_ok=True)
    ok = ~z['overlap']
    N = int(ok.sum())
    lay = z['layout'][ok].astype(np.int64)
    clus = lay * 441 + z['o'][ok].astype(np.int64)
    _, clus = np.unique(clus, return_inverse=True)
    res = {'n_placements': N, 'n_clusters': int(clus.max() + 1), 'encoders_identical_all_24': chk['encoders_identical_all_24']}

    # ---------------- d (pixel differences)
    DK = ('h2_v2', 'h2_v1', 'h1_v1', 'h1_v2', 'h2_v2n', 'h2_v1n', 'h1_v1n', 'h1_v2n')
    dv = {k: z['d_' + k][ok].astype(np.float64) for k in DK}
    dv['delta'] = dv['h2_v1'] - dv['h2_v2']                # > 0: v2hat closer to A-2's view than to A-1's view
    dv['delta_n'] = dv['h2_v1n'] - dv['h2_v2n']
    dkeys = list(dv)
    Dmat = np.stack([dv[k] for k in dkeys], 1)
    # ---------------- per-placement correlation statistics: v2hat vs v2 and v1hat vs v1, per measure / range set
    qs = {}
    for mk in MEAS:
        for bn, bins in BINSETS.items():
            qs[f'h2v2|{mk}|{bn}'] = stats(z[f'h2_{mk}'][ok], z[f'v2_{mk}'][ok], bins)
            qs[f'h1v1|{mk}|{bn}'] = stats(z[f'h1_{mk}'][ok], z[f'v1_{mk}'][ok], bins)
    # read-outs (test half)
    te = rd['test_idx']
    te_ok = np.searchsorted(np.where(ok)[0], te)       # positions of the test placements among the non-overlap ones
    assert np.array_equal(np.where(ok)[0][te_ok], te)
    rq = {}
    for fn in ('F2', 'F1'):
        for tn in ('v2_col_thr', 'v2_pos_thr', 'v2_col_cont', 'v2_pos_cont'):
            for bn, bins in BINSETS.items():
                rq[f'{fn}|{tn}|{bn}'] = stats(rd[f'{fn}|{tn}'], z[tn][te], bins)
    qk, rk = list(qs), list(rq)
    Q = np.concatenate([qs[k] for k in qk], 1)          # (N, 7 * len(qk))
    RQ = np.concatenate([rq[k] for k in rk], 1)         # (Ntest, 7 * len(rk))

    def point():
        out = {'d': {k: float(dv[k].mean()) for k in dkeys}}
        r, r2 = r_r2(Q.sum(0).reshape(-1, 7))
        out['corr'] = {k: float(r[i]) for i, k in enumerate(qk)}
        r, r2 = r_r2(RQ.sum(0).reshape(-1, 7))
        out['read_r'] = {k: float(r[i]) for i, k in enumerate(rk)}
        out['read_R2'] = {k: float(r2[i]) for i, k in enumerate(rk)}
        return out

    P = point()
    # bootstraps
    B = {}
    for name, cl in (('boot', None), ('boot_cluster', clus)):
        rs = np.random.RandomState(0)
        bd, bc = [], []
        for W in boot_counts(N, NB, rs, cl):
            bd.append((W @ Dmat) / W.sum(1, keepdims=True))
            bc.append(r_r2((W @ Q).reshape(W.shape[0], -1, 7))[0])
        bd, bc = np.concatenate(bd), np.concatenate(bc)
        rs = np.random.RandomState(0)
        Nt = len(te)
        br, br2 = [], []
        cl_te = None if cl is None else np.unique(cl[te_ok], return_inverse=True)[1]
        for W in boot_counts(Nt, NB, rs, cl_te):
            a, b2 = r_r2((W @ RQ).reshape(W.shape[0], -1, 7))
            br.append(a)
            br2.append(b2)
        br, br2 = np.concatenate(br), np.concatenate(br2)
        B[name] = {'d': {k: ci(bd[:, i]) for i, k in enumerate(dkeys)},
                   'corr': {k: ci(bc[:, i]) for i, k in enumerate(qk)},
                   'read_r': {k: ci(br[:, i]) for i, k in enumerate(rk)},
                   'read_R2': {k: ci(br2[:, i]) for i, k in enumerate(rk)}}
    res['point'] = P
    res.update(B)
    res['pair_t_ci'] = 'zero width: the decoded images and read-outs are the same for every pair and condition (Encoders identical in all 24 predictors)'
    # descriptive: per layout correlations (point)
    per = {}
    for li, ln in enumerate(('original', 'swap_rg')):
        m = lay == li
        for k in qk:
            if k.endswith(('c0167', 'all')):
                per[f'{ln}|{k}'] = float(r_r2(qs[k][m].sum(0))[0])
        mt = lay[te_ok] == li
        for k in rk:
            if k.endswith(('c0167', 'all')):
                per[f'{ln}|read|{k}'] = float(r_r2(rq[k][mt].sum(0))[0])
        for k in ('h2_v2', 'h2_v1', 'h1_v1', 'h2_v2n', 'h2_v1n', 'h1_v1n'):
            per[f'{ln}|d|{k}'] = float(dv[k][m].mean())
    res['per_layout'] = per
    # reproduction of the v12 Fig.5 json (original layout, overlap included, unclamped, no-agent views)
    j = json.load(open(V12_JSON))['original']
    m0 = z['layout'] == 0
    res['repro_v12_json'] = {
        'other_other': [float(z['d_raw_h2_v2n'][m0].mean()), j['other_other']['mean']],
        'other_self': [float(z['d_raw_h1_v2n'][m0].mean()), j['other_self']['mean']],
        'self_self': [float(z['d_raw_h1_v1n'][m0].mean()), j['self_self']['mean']],
        'self_other': [float(z['d_raw_h2_v1n'][m0].mean()), j['self_other']['mean']]}
    res['orig_exp2_result_txt'] = ORIG_EXP2

    # ---------------- reading-table row (p50; the same for every condition)
    def val(k):
        v = P['corr'][k]
        return v if np.isfinite(v) else -np.inf
    lo = B['boot']['d']['delta'][0]
    c_d = bool(P['d']['delta'] > 0 and lo is not None and lo > 0)
    c_pos = bool(max(val('h2v2|pos_thr|c0167'), val('h2v2|pos_cont|c0167')) >= 0.7)
    c_col = bool(max(val('h2v2|col_thr|c0167'), val('h2v2|col_cont|c0167')) >= 0.7)
    rr = P['read_r']['F2|v2_col_thr|c0167']
    c_read = bool(np.isfinite(rr) and rr >= 0.7)
    if not c_d or not c_pos:
        row = 4
    elif c_col:
        row = 1
    else:
        row = 2 if c_read else 3
    res['table'] = {'d_less': c_d, 'pos_high': c_pos, 'col_high': c_col, 'read_col_high': c_read, 'row': row}
    json.dump(res, open(os.path.join(od, 'summary.json'), 'w'), indent=1)
    tables(res, od)
    figs(z, od)
    print(json.dumps(res['table']), json.dumps(res['point']['d']))


def f3(x):
    return 'undefined' if x is None or not np.isfinite(x) else f'{x:.3f}'


def cis(c):
    return '[' + ', '.join('—' if v is None else f'{v:.3f}' for v in c) + ']'


def tables(res, od):
    P, B, C = res['point'], res['boot'], res['boot_cluster']
    L = ['# 第31弾 集計の表（`round31_aggregate.py`）', '',
         f"- 配置 {res['n_placements']}（重なりを除く、2 配置 × 361 × 360）。エンコーダは 24 本の予測器で同一 → 条件・組によらず同じ値。6 組の t の CI は幅 0。",
         '- CI：ブートストラップ（組 → 配置、1000 回）/ 記述：A-2 の位置ごとのまとまり（722）のブートストラップ。', '',
         '## §2.3-1 画素の差 d（[0,1] の画素の差の絶対値の平均）', '',
         '| 量 | 値 | CI（配置） | CI（まとまり） |', '|---|---|---|---|']
    names = {'h2_v2': 'd(v̂², v²)', 'h2_v1': 'd(v̂², v¹)', 'h1_v1': 'd(v̂¹, v¹)', 'h1_v2': 'd(v̂¹, v²)',
             'delta': 'd(v̂², v¹) − d(v̂², v²)', 'h2_v2n': 'd(v̂², v²) 体なし', 'h2_v1n': 'd(v̂², v¹) 体なし',
             'h1_v1n': 'd(v̂¹, v¹) 体なし', 'h1_v2n': 'd(v̂¹, v²) 体なし', 'delta_n': '差 体なし'}
    for k, nm in names.items():
        L.append(f"| {nm} | {P['d'][k]:.4f} | {cis(B['d'][k])} | {cis(C['d'][k])} |")
    L += ['', '元のコードの出力（exp2、体なしの視界）：' + '、'.join(f'{k} {v:.4f}' for k, v in res['orig_exp2_result_txt'].items()), '',
          'v12 の Fig.5 の json との再現（元の配置、重なりを含む、クリップなし、体なし）：' +
          '、'.join(f'{k} {a:.6f} / {b:.6f}' for k, (a, b) in res['repro_v12_json'].items()), '',
          '## §2.3-2・3 位置の量・色の量（v̂² と v² の相関。参考：v̂¹ と v¹）', '',
          '| 量 | 範囲 | v̂²–v² r | CI（配置） | CI（まとまり） | v̂¹–v¹ r |', '|---|---|---|---|---|---|']
    for mk in MEAS:
        for bn in BINSETS:
            k = f'h2v2|{mk}|{bn}'
            L.append(f"| {mk} | {bn} | {f3(P['corr'][k])} | {cis(B['corr'][k])} | {cis(C['corr'][k])} | {f3(P['corr'][f'h1v1|{mk}|{bn}'])} |")
    L += ['', '## §2.4 線形の読み出し（試しの半分）', '',
          '| 特徴 → 的 | 範囲 | r | CI（配置） | CI（まとまり） | R² | CI（配置） |', '|---|---|---|---|---|---|---|']
    for k in P['read_r']:
        if k.endswith(('c0167', 'all')):
            L.append(f"| {k.rsplit('|', 1)[0]} | {k.rsplit('|', 1)[1]} | {f3(P['read_r'][k])} | {cis(B['read_r'][k])} | {cis(C['read_r'][k])} | "
                     f"{f3(P['read_R2'][k])} | {cis(B['read_R2'][k])} |")
    L += ['', '## 範囲ごとの読み出しの r（F2 → v2_col_thr）', '', '| 範囲 | r | R² |', '|---|---|---|']
    for b in range(8):
        k = f'F2|v2_col_thr|b{b}'
        L.append(f"| {b} | {f3(P['read_r'][k])} | {f3(P['read_R2'][k])} |")
    L += ['', '## 配置ごと（記述）', '', '| 項目 | 値 |', '|---|---|'] + [f'| {k} | {f3(v)} |' for k, v in res['per_layout'].items()]
    L += ['', '## 読み方の表', '', f"- {json.dumps(res['table'], ensure_ascii=False)}"]
    open(os.path.join(od, 'tables.md'), 'w').write('\n'.join(L) + '\n')


def figs(z, od):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ex = np.load(os.path.join(OUT, 'main', 'examples.npz'))
    keys = [k for k in ex.files]
    sel = []
    for lay in ('original', 'swap_rg'):
        for a1, a2 in (((-5, 5), (5, -5)), ((5, 5), (-5, -5)), ((0, -7), (0, 7)), ((-7, 0), (7, 0))):
            k = f'{lay}|A1{a1}|A2{a2}'
            if k in keys:
                sel.append(k)
    fig, ax = plt.subplots(len(sel), 4, figsize=(13, 0.95 * len(sel) + 0.6))
    for i, k in enumerate(sel):
        for j, t in enumerate(('v¹ (A-1 view)', 'v² (A-2 view)', 'v̂¹ = D(E1(v¹))', 'v̂² = D(E2(v¹))')):
            ax[i, j].imshow(ex[k][j], aspect='auto', interpolation='nearest')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(t, fontsize=9)
        ax[i, 0].set_ylabel(k.replace('|', '\n'), fontsize=6, rotation=0, ha='right', va='center')
    fig.suptitle('Round 31: original-paper decoding on the grid (existing Visual Decoder v12_fig5_s0); array channels as RGB', fontsize=9)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(od, f'fig_round31_examples.{e}'), dpi=130)
    plt.close(fig)
    s = json.load(open(os.path.join(od, 'summary.json')))
    P, B = s['point']['d'], s['boot']['d']
    ks = ['h1_v1', 'h2_v2', 'h2_v1', 'h1_v2']
    lab = ['d(v̂¹,v¹)', 'd(v̂²,v²)', 'd(v̂²,v¹)', 'd(v̂¹,v²)']
    orig = [s['orig_exp2_result_txt'][k] for k in ('self_self', 'other_other', 'self_other', 'other_self')]
    x = np.arange(4)
    fig, a = plt.subplots(figsize=(7, 3.4))
    a.bar(x - 0.27, [P[k] for k in ks], 0.27, label='round 31 (views with agents)', color='#4c78a8')
    a.bar(x, [P[k + 'n'] for k in ks], 0.27, label='round 31 (agent-free views, as original code)', color='#9ecae9')
    a.bar(x + 0.27, orig, 0.27, label='original exp2 result.txt (agent-free)', color='#bbbbbb')
    a.set_xticks(x)
    a.set_xticklabels(lab)
    a.set_ylabel('mean |pixel difference|')
    a.legend(fontsize=7)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(od, f'fig_round31_d.{e}'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
