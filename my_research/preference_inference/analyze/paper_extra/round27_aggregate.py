"""
Paper round 27 aggregation (docs/paper_extra_log.md Sec.51.2; definitions fixed there before computing).
  sig  Sec.1: S_WT = L(W) - L(T), S_ZT = L(Z) - L(T) (window sums) by component (total, L1, L1_A2, L1_rest,
       feat_other, feat_self) x A-2 liking (all, red, green) x flow (V, B) x group (r25 swap / no swap, r20 O / S) x
       condition (none / p50 / sr50 / p80; p50 - sr50, p50 - none); relative = S / mean L(T) of the pair, group,
       liking and component; 6-pair t CI + two-stage bootstrap (round21_aggregate.stat); readings of the Sec.1 table;
       swap-unrelated check. Also checks that the recomputed L1 / features equal round 26's saved values (all files).
       -> Round27/analysis/sig_{summary.json,tables.md}, fig_round27_sig.{pdf,png}
  s2   Sec.2: D_T(e) = C(T) - C(Z) per condition x pair (all, red, green; episode bootstrap), D_T,late (epochs 40-50)
       3-pair t CI + two-stage bootstrap, readings; C_self and L1 (V, B) curves; S_WT (B) by component x liking at
       epochs 0 and 50 (r25 train main, test reference) and 50 - 0. -> Round27/analysis/s2_{summary.json,tables.md},
       fig_round27_s2.{pdf,png}
Usage: python analyze/paper_extra/round27_aggregate.py sig|s2
"""

import glob
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round21_aggregate as A  # noqa  (stat)
import vdir_stageA as VA  # noqa  (tsum)

_PI = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(_PI, 'data', 'result')
R27 = os.path.join(RES, 'paper_extra', 'Round27')
R26 = os.path.join(RES, 'paper_extra', 'Round26')
D25 = os.path.join(_PI, 'data', 'data', 'paper_r25_swap50', 'data.h5')
AN = os.path.join(R27, 'analysis')
CONDS = ('p50', 'sr50', 'none', 'p80')
CN = {'p50': '経験50%', 'sr50': '符号無作為（経験50%）', 'none': '経験なし', 'p80': '経験80%（記述）',
      'p50-sr50': '経験50% − 符号無作為', 'p50-none': '経験50% − 経験なし'}
COMP = ('total', 'l1', 'l1_a2', 'l1_rest', 'feat_other', 'feat_self')
CMN = {'total': '全体 $L$', 'l1': 'A-1 の視界の L1', 'l1_a2': 'L1：A-2 の周り', 'l1_rest': 'L1：それ以外',
       'feat_other': '他者の特徴', 'feat_self': 'そのほか（自己の特徴）'}
LIKES = ('all', 'red', 'green')
LN = {'all': '全部', 'red': '赤好き', 'green': '緑好き'}
GROUPS = ('r25_swap', 'r25_noswap', 'r20_O', 'r20_S')
GN = {'r25_swap': '第25弾・入れ替えあり', 'r25_noswap': '第25弾・入れ替えなし', 'r20_O': '第20弾 O（入れ替えあり）',
      'r20_S': '第20弾 S（入れ替えなし）'}


def cls(st):
    t, b = st['t']['ci95'], st['boot']
    if t is None:
        return '—'
    if t[0] > 0 and b[0] > 0:
        return '>0'
    if t[1] < 0 and b[1] < 0:
        return '<0'
    if t[0] <= 0 <= t[1] and b[0] <= 0 <= b[1]:
        return '≈0'
    return '食い違う'


def fmt(st, d=4):
    t, b = st['t']['ci95'], st['boot']
    return f"{st['mean']:+.{d}f} [{t[0]:+.{d}f}, {t[1]:+.{d}f}] [{b[0]:+.{d}f}, {b[1]:+.{d}f}]"


def comps(a):
    """a (..., 4) = [l1_a2, l1_rest, feat_self, feat_other] -> (..., 6) in COMP order."""
    l1 = a[..., 0] + a[..., 1]
    return np.stack([l1 + a[..., 2] + a[..., 3], l1, a[..., 0], a[..., 1], a[..., 3], a[..., 2]], -1)


def like_mask(liked, lk):
    return np.ones(len(liked), bool) if lk == 'all' else (liked == (1 if lk == 'red' else 0))


def diffs_from(L, liked):
    """L: {inp: (npairs, N, 6)} for one condition / group / flow; liked (N,). -> {(q, comp, like): (npairs, N)}."""
    out = {}
    for lk in LIKES:
        mk = like_mask(liked, lk)
        for q, (a, b) in (('S_WT', ('W', 'T')), ('S_ZT', ('Z', 'T'))):
            d = L[a] - L[b]
            for k, comp in enumerate(COMP):
                x = np.where(mk[None], d[..., k], np.nan)
                lt = np.where(mk[None], L['T'][..., k], np.nan)
                out[(q, comp, lk)] = x
                with np.errstate(all='ignore'):
                    out[(q + '_rel', comp, lk)] = x / np.nanmean(lt, 1, keepdims=True)
        for k, comp in enumerate(COMP):
            out[('L_T', comp, lk)] = np.where(mk[None], L['T'][..., k], np.nan)
    return out


# ---------------------------------------------------------------- Sec.1
def sig_arrays():
    Z = {c: [np.load(os.path.join(R27, 'signal', f'{c}_s{s}.npz')) for s in range(6)] for c in CONDS}
    sw = Z['p50'][0]['r25_swap'].astype(bool)
    sel = {'r25_swap': ('r25', sw), 'r25_noswap': ('r25', ~sw), 'r20_O': ('r20O', None), 'r20_S': ('r20S', None)}
    X = {}
    for c in CONDS:
        for g, (pre, m) in sel.items():
            liked = Z[c][0][f'{pre}_liked'] if pre != 'r25' else Z[c][0]['r25_liked']
            liked = liked if m is None else liked[m]
            for f in ('V', 'B'):
                L = {}
                for i in ('T', 'W', 'Z'):
                    a = np.stack([z[f'{pre}_{f}_{i}'] for z in Z[c]]).astype(np.float64)   # (6, N, 4)
                    L[i] = comps(a if m is None else a[:, m])
                for (q, comp, lk), v in diffs_from(L, liked).items():
                    X[(c, g, f, q, comp, lk)] = v
            rs = np.stack([z[f'{pre}_region_share'] for z in Z[c]])
            X[(c, g, 'V', 'region_share', 'l1_a2', 'all')] = (rs if m is None else rs[:, m]) / 20.0
    return X, Z


def reproduce_r26(Z):
    """Recomputed [L1, feat_self, feat_other] vs round 26's saved window sums (all conditions, pairs, groups)."""
    mx = 0.0
    for c in CONDS:
        for s in range(6):
            z26 = np.load(os.path.join(R26, 'signal', f'{c}_s{s}.npz'))
            for pre in ('r25', 'r20S', 'r20O'):
                for f in ('V', 'B'):
                    for i in ('T', 'W', 'Z'):
                        n_ = Z[c][s][f'{pre}_{f}_{i}'].astype(np.float64)
                        mine = np.stack([n_[:, 0] + n_[:, 1], n_[:, 2], n_[:, 3]], 1)
                        o = z26[f'{pre}_{f}_{i}'].astype(np.float64)
                        mx = max(mx, float(np.max(np.abs(mine - o) / np.maximum(np.abs(o), 1e-6))))
    return mx


def sig():
    os.makedirs(AN, exist_ok=True)
    X, Z = sig_arrays()
    rep = reproduce_r26(Z)
    S = {}
    for key, a in X.items():
        if key[3] in ('L_T', 'region_share'):
            with np.errstate(all='ignore'):
                S['|'.join(key)] = {'mean': float(np.nanmean(np.nanmean(a, 1)))}
        else:
            S['|'.join(key)] = A.stat(a)
    for o in ('sr50', 'none'):
        for g in GROUPS:
            for f in ('V', 'B'):
                for q in ('S_WT', 'S_ZT'):
                    for comp in COMP:
                        for lk in LIKES:
                            S['|'.join((f'p50-{o}', g, f, q, comp, lk))] = A.stat(X[('p50', g, f, q, comp, lk)] -
                                                                               X[(o, g, f, q, comp, lk)])
    k = lambda c, g, f, comp, lk, q='S_WT': S[f'{c}|{g}|{f}|{q}|{comp}|{lk}']  # noqa
    # readings (main: p50, r25 swap, flow B)
    red, green = cls(k('p50', 'r25_swap', 'B', 'total', 'red')), cls(k('p50', 'r25_swap', 'B', 'total', 'green'))
    pref_row = 1 if (red == '>0' and green in ('≈0', '<0')) else (2 if (red == '>0' and green == '>0') else None)
    comp_rows = {}
    for comp in ('l1_a2', 'l1'):
        cc = cls(k('p50', 'r25_swap', 'B', comp, 'all'))
        comp_rows[comp] = {'cls': cc, 'row': {'≈0': 3, '>0': 4}.get(cc)}
    rd = {'preference': {'red': red, 'green': green, 'row': pref_row}, 'component': comp_rows}
    # swap-unrelated gain (round 26 rule) for every condition / flow / component / liking
    nos = {}
    for c in CONDS:
        for f in ('V', 'B'):
            for comp in COMP:
                for lk in LIKES:
                    with np.errstate(all='ignore'):
                        ps = np.nanmean(X[(c, 'r25_swap', f, 'S_WT', comp, lk)], 1)
                        pn = np.nanmean(X[(c, 'r25_noswap', f, 'S_WT', comp, lk)], 1)
                    t = VA.tsum((ps - pn).tolist())
                    ncl = cls(k(c, 'r25_noswap', f, comp, lk))
                    nos[f'{c}|{f}|{comp}|{lk}'] = {'swap_minus_noswap': t, 'noswap_cls': ncl,
                                                    'unrelated_to_swap': bool(ncl == '>0' and t['ci95'] is not None and
                                                                              t['ci95'][0] <= 0 <= t['ci95'][1])}
    json.dump({'stats': S, 'reading': rd, 'noswap': nos, 'reproduce_round26_max_rel_diff': rep,
               'sec': {f'{c}_s{s}': float(Z[c][s]['sec']) for c in CONDS for s in range(6)}},
              open(os.path.join(AN, 'sig_summary.json'), 'w'), indent=1, ensure_ascii=False)
    L = ['# 第27弾 §1 正しい好みの得の内訳', '',
         '値：平均 [6 組の t の 95% CI] [2段のブートストラップの 95% CI]。窓（入れ替えから 20 ステップ）の合計。'
         '判定：>0・≈0・<0・食い違う（2 つの CI で、第26弾と同じ）。', '',
         f'- 第26弾の保存した値との一致（全条件・全組・全群、L1・自己の特徴・他者の特徴）：相対の差の最大 {rep:.2e}。',
         f"- A-2 の周りの領域の大きさ（1 ステップあたりの列の割合、第25弾・入れ替えあり、経験50%）：{S['p50|r25_swap|V|region_share|l1_a2|all']['mean']:.3f}。", '',
         '## 読み方の表の当てはめ（主：経験50%、第25弾・入れ替えあり、B）', '', '```',
         json.dumps(rd, ensure_ascii=False, indent=1), '```', '',
         '## 主の表：経験50%・第25弾・入れ替えあり、$S_{\\mathrm{WT}}$（成分 × 好み）', '']
    for f in ('B', 'V'):
        L += [f'### 流し方 {f}', '', '| 成分 | 全部 | 赤好き | 緑好き | 判定（全部／赤／緑） |', '|---|---|---|---|---|']
        for comp in COMP:
            sts = [k('p50', 'r25_swap', f, comp, lk) for lk in LIKES]
            L.append(f"| {CMN[comp]} | " + ' | '.join(fmt(s_) for s_ in sts) + ' | ' + '／'.join(cls(s_) for s_ in sts) + ' |')
        L.append('')
    L += ['## すべての条件・群・流し方：$S_{\\mathrm{WT}}$（全部・赤好き・緑好き）', '']
    for comp in COMP:
        L += [f'### {CMN[comp]}', '', '| 条件・差 | 群 | 流し方 | 全部 | 赤好き | 緑好き | 判定 |', '|---|---|---|---|---|---|---|']
        for c in CONDS + ('p50-sr50', 'p50-none'):
            for g in GROUPS:
                for f in ('V', 'B'):
                    sts = [k(c, g, f, comp, lk) for lk in LIKES]
                    L.append(f'| {CN[c]} | {GN[g]} | {f} | ' + ' | '.join(fmt(s_) for s_ in sts) + ' | ' +
                             '／'.join(cls(s_) for s_ in sts) + ' |')
        L.append('')
    L += ['## 相対の得（$S_{\\mathrm{WT}}$ ÷ その成分の $L(T)$ の平均）と $L(T)$（第25弾・入れ替えあり）', '',
          '| 条件 | 流し方 | 成分 | 相対 全部 | 相対 赤好き | 相対 緑好き | $L(T)$ 全部 |', '|---|---|---|---|---|---|---|']
    for c in CONDS:
        for f in ('V', 'B'):
            for comp in COMP[:5]:
                L.append(f'| {CN[c]} | {f} | {CMN[comp]} | ' + ' | '.join(fmt(k(c, 'r25_swap', f, comp, lk, 'S_WT_rel'))
                                                                     for lk in LIKES) +
                         f" | {S[f'{c}|r25_swap|{f}|L_T|{comp}|all']['mean']:.4f} |")
    L += ['', '## $S_{\\mathrm{ZT}}$ と相対（第25弾・入れ替えあり、全部）', '',
          '| 条件 | 流し方 | 成分 | $S_{\\mathrm{ZT}}$ | 相対 | 判定 |', '|---|---|---|---|---|---|']
    for c in CONDS:
        for f in ('V', 'B'):
            for comp in COMP[:5]:
                s_ = k(c, 'r25_swap', f, comp, 'all', 'S_ZT')
                L.append(f"| {CN[c]} | {f} | {CMN[comp]} | {fmt(s_)} | {fmt(k(c, 'r25_swap', f, comp, 'all', 'S_ZT_rel'))} | {cls(s_)} |")
    L += ['', '## $S_{\\mathrm{ZT}}$ の好み別（経験50%、第25弾・入れ替えあり）', '',
          '| 流し方 | 成分 | 全部 | 赤好き | 緑好き |', '|---|---|---|---|---|']
    for f in ('B', 'V'):
        for comp in COMP[:5]:
            L.append(f'| {f} | {CMN[comp]} | ' + ' | '.join(fmt(k('p50', 'r25_swap', f, comp, lk, 'S_ZT')) for lk in LIKES) + ' |')
    L += ['', '## 入れ替えと関係のない得か（第26弾の決まり。「はい」になったものだけ）', '']
    hits = [kk for kk, v in nos.items() if v['unrelated_to_swap']]
    L += [f'- {kk}' for kk in hits] or ['- なし']
    open(os.path.join(AN, 'sig_tables.md'), 'w').write('\n'.join(L) + '\n')
    fig_sig(X)
    print('\n'.join(L[:60]))


def fig_sig(X):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'all': 'k', 'red': 'tab:red', 'green': 'tab:green'}
    panels = (('r25_swap', 'B'), ('r25_swap', 'V'), ('r25_noswap', 'B'))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (g, f) in zip(axes, panels):
        for ci, comp in enumerate(COMP):
            for li, lk in enumerate(LIKES):
                xpos = ci * 4 + li
                with np.errstate(all='ignore'):
                    pm = np.nanmean(X[('p50', g, f, 'S_WT', comp, lk)], 1)
                ax.scatter(np.full(6, xpos), pm, s=14, color=col[lk], alpha=0.6)
                half = 2.571 * pm.std(ddof=1) / np.sqrt(6)
                ax.errorbar(xpos, pm.mean(), yerr=half, fmt='_', color=col[lk], ms=14, capsize=3)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks([ci * 4 + 1 for ci in range(len(COMP))])
        ax.set_xticklabels(['total', 'L1', 'L1 A-2', 'L1 rest', 'feat other', 'feat self'], fontsize=8)
        ax.set_title(f'p50, {g}, flow {f}: S_WT (black all, red red-liker, green green-liker)', fontsize=9)
    fig.suptitle('round 27 Sec.1: S_WT = L(W) - L(T) by component x A-2 liking; dots = pairs, bar = mean +- 6-pair t CI')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(AN, f'fig_round27_sig.{ext}'), dpi=110)


# ---------------------------------------------------------------- Sec.2
def boot_ep(x, rng_seed=0, B=1000):
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return {'mean': float(x.mean()), 'ci': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], 'n': int(len(x))}


def d_of(z, liked=None):
    d = (z['S_T_delta'] - z['O_T_delta']) - (z['S_Z_delta'] - z['O_Z_delta'])
    if liked is not None:
        d = np.where(z['a2_liked'] == liked, d, np.nan)
    return d


def mv(st):
    return '>0' if st['ci'][0] > 0 else ('<0' if st['ci'][1] < 0 else '≈0')


def s2_runs():
    runs = {}
    for p in sorted(glob.glob(os.path.join(R27, 'runs', 's2_*'))):
        _, c, s = os.path.basename(p).split('_')
        ev = {int(os.path.basename(e)[2:5]): np.load(e) for e in sorted(glob.glob(os.path.join(p, 'eval', 'ep*.npz')))}
        sg = {int(os.path.basename(e)[6:9]): np.load(e) for e in sorted(glob.glob(os.path.join(p, 'eval', 'sig_ep*.npz')))}
        runs[(c, int(s[1:]))] = {'ev': ev, 'sig': sg, 'log': json.load(open(os.path.join(p, 'log.json'))),
                                 'done': os.path.exists(os.path.join(p, 'done.json'))}
    return runs


def s2():
    os.makedirs(AN, exist_ok=True)
    runs = s2_runs()
    pairs = sorted({s for (_, s) in runs})
    conds = [c for c in ('p50', 'sr50', 'none') if all((c, s) in runs for s in pairs)]
    S, curves, mon = {}, {}, {}
    late = {}
    for (c, s), r in runs.items():
        for lk, lv in (('all', None), ('red', 1), ('green', 0)):
            curves[f'{c}|{s}|{lk}'] = {e: boot_ep(d_of(z, lv)) for e, z in r['ev'].items()}
            if all(e in r['ev'] for e in (40, 45, 50)):
                lt = np.nanmean(np.stack([d_of(r['ev'][e], lv) for e in (40, 45, 50)]), 0)
                late[(c, s, lk)] = lt
                S[f'{c}|{s}|{lk}|late'] = boot_ep(lt)
            S[f'{c}|{s}|{lk}|ep0'] = boot_ep(d_of(r['ev'][0], lv))
        mon[f'{c}|{s}'] = {str(e): {'c_self': float(np.nanmean(z['c_self'])), 'l1_T_V': float(z['l1_T_V']),
                                    'l1_T_B': float(z['l1_T_B']),
                                    'readable_min': float(min(float(z[q]) for q in z.files if q.endswith('_readable')))}
                           for e, z in r['ev'].items()}
    multi = {}
    for c in conds:
        for lk in LIKES:
            if all((c, s, lk) in late for s in pairs):
                multi[f'{c}|{lk}'] = A.stat(np.stack([late[(c, s, lk)] for s in pairs]))
    for o in ('sr50', 'none'):
        if o in conds and 'p50' in conds:
            multi[f'p50-{o}|all'] = A.stat(np.stack([late[('p50', s, 'all')] - late[(o, s, 'all')] for s in pairs]))
            for s in pairs:
                S[f'p50-{o}|{s}|all|late'] = boot_ep(late[('p50', s, 'all')] - late[(o, s, 'all')])
    # readings
    rd = {}
    if all(f'{c}|all' in multi for c in ('p50', 'sr50', 'none')):
        cl = {c: cls(multi[f'{c}|all']) for c in ('p50', 'sr50', 'none')}
        any_pos = any(v == '>0' for v in cl.values())
        rd = {'cls': cl, 'row1_p50_pos': cl['p50'] == '>0',
              'row2_no_condition_pos': not any_pos, 'row2_all_approx0': all(v == '≈0' for v in cl.values()),
              'row3_only_none_or_sr50_pos': cl['p50'] != '>0' and any_pos,
              'break_rule': not any_pos}
        rows4 = {}
        for (c, s), r in runs.items():
            if 0 in r['ev'] and 50 in r['ev']:
                c0, c50 = float(np.nanmean(r['ev'][0]['c_self'])), float(np.nanmean(r['ev'][50]['c_self']))
                rows4[f'{c}|{s}'] = {'c_self_0': c0, 'c_self_50': c50, 'ratio': c50 / c0, 'below_half': c50 < 0.5 * c0}
        rd['row4_c_self'] = rows4
    # S_WT (flow B) at epochs 0 and 50 by component x liking
    g = h5py.File(D25, 'r')
    meta = {tag: (g[sp]['swap_frame'][()] >= 0, g[sp]['a2_liked'][()].astype(np.int64)) for tag, sp in (('tr', 'train'), ('te', 'test'))}
    SW = {}
    for tag in ('tr', 'te'):
        swm, liked = meta[tag]
        for gname, m in (('swap', swm), ('noswap', ~swm)):
            for c in conds:
                arr = {}
                for e in (0, 50):
                    if not all(e in runs[(c, s)]['sig'] for s in pairs):
                        continue
                    L = {i: np.stack([comps(runs[(c, s)]['sig'][e][f'{tag}_B_{i}'].astype(np.float64))[m] for s in pairs])
                         for i in ('T', 'W', 'Z')}
                    arr[e] = diffs_from(L, liked[m])
                for e, dd in arr.items():
                    for (q, comp, lk), v in dd.items():
                        if q in ('S_WT', 'S_ZT', 'S_WT_rel'):
                            SW[f'{tag}|{gname}|{c}|ep{e}|{q}|{comp}|{lk}'] = A.stat(v)
                if 0 in arr and 50 in arr:
                    for comp in COMP:
                        for lk in LIKES:
                            SW[f'{tag}|{gname}|{c}|ep50-ep0|S_WT|{comp}|{lk}'] = A.stat(arr[50][('S_WT', comp, lk)] -
                                                                                      arr[0][('S_WT', comp, lk)])
    json.dump({'stats': S, 'multi': multi, 'reading': rd, 'monitor': mon, 'swt': SW,
               'curves': {k: {str(e): v for e, v in d.items()} for k, d in curves.items()},
               'runs': {f'{c}|{s}': {'done': r['done'], 'last_epoch': max(r['ev']),
                                     'epoch_sec_median': float(np.median([x['sec'] for x in r['log']['epochs']])) if r['log']['epochs'] else None,
                                     'adapter_w_last': r['ev'][max(r['ev'])]['adapter_w'].round(3).tolist()}
                        for (c, s), r in runs.items()}},
              open(os.path.join(AN, 's2_summary.json'), 'w'), indent=1, ensure_ascii=False)
    f3 = lambda st: f"{st['mean']:+.4f} [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}]"  # noqa
    L = ['# 第27弾 §2 最も緩い上限（T、B、足した層 + 共有の LSTM、段階Iのデータを混ぜる）', '',
         '値（組ごと）：平均 [エピソードのブートストラップの 95% CI]。3 組：平均 [3 組の t CI] [2段のブートストラップ]。'
         '$D_{T,late}$ = 40・45・50 エポックの平均。', '',
         '## 読み方の表の当てはめ', '', '```', json.dumps(rd, ensure_ascii=False, indent=1), '```', '',
         f'## 3 組（組 {", ".join(str(p + 1) for p in pairs)}）をまとめた $D_{{T,late}}$', '',
         '| 条件・差 | 全部 | 赤好き | 緑好き | 判定（全部） |', '|---|---|---|---|---|']
    for c in conds + [f'p50-{o}' for o in ('sr50', 'none') if f'p50-{o}|all' in multi]:
        row = [fmt(multi[f'{c}|{lk}']) if f'{c}|{lk}' in multi else '—' for lk in LIKES]
        L.append(f"| {CN[c]} | " + ' | '.join(row) + f" | {cls(multi[f'{c}|all'])} |")
    L += ['', '## 組ごとの $D_T$（0 エポック目と late）', '',
          '| 組 | 条件・差 | 0 エポック目（全部） | late 全部 | late 赤好き | late 緑好き | 判定 |', '|---|---|---|---|---|---|---|']
    for s in pairs:
        for c in conds:
            if f'{c}|{s}|all|late' in S:
                L.append(f"| {s + 1} | {CN[c]} | {f3(S[f'{c}|{s}|all|ep0'])} | {f3(S[f'{c}|{s}|all|late'])} | "
                         f"{f3(S[f'{c}|{s}|red|late'])} | {f3(S[f'{c}|{s}|green|late'])} | {mv(S[f'{c}|{s}|all|late'])} |")
        for o in ('sr50', 'none'):
            kk = f'p50-{o}|{s}|all|late'
            if kk in S:
                L.append(f"| {s + 1} | {CN['p50-' + o]} | — | {f3(S[kk])} | — | — | {mv(S[kk])} |")
    eps = sorted({e for r in runs.values() for e in r['ev']})
    L += ['', '## $D_T(e)$（全部）・$C_{self}$・A-1 の視界の L1（V／B）', '',
          '| 走らせたもの | 量 | ' + ' | '.join(f'ep{e}' for e in eps) + ' |', '|---|---|' + '---|' * len(eps)]
    for (c, s) in sorted(runs):
        cv = curves[f'{c}|{s}|all']
        L.append(f'| {c}_s{s} | $D_T$ | ' + ' | '.join(f"{cv[e]['mean']:+.3f}" if e in cv else '' for e in eps) + ' |')
        mm = mon[f'{c}|{s}']
        L.append(f'| {c}_s{s} | $C_{{self}}$ | ' + ' | '.join(f"{mm[str(e)]['c_self']:+.3f}" if str(e) in mm else '' for e in eps) + ' |')
        L.append(f'| {c}_s{s} | L1 V | ' + ' | '.join(f"{mm[str(e)]['l1_T_V']:.4f}" if str(e) in mm else '' for e in eps) + ' |')
        L.append(f'| {c}_s{s} | L1 B | ' + ' | '.join(f"{mm[str(e)]['l1_T_B']:.4f}" if str(e) in mm else '' for e in eps) + ' |')
    L += ['', '## 0・50 エポック目の $S_{\\mathrm{WT}}$（B、第25弾の学習用データ・入れ替えあり、3 組）', '',
          '| 条件 | 成分 | 好み | 0 エポック目 | 50 エポック目 | 50 − 0 | 判定（0／50／差） |', '|---|---|---|---|---|---|---|']
    for c in conds:
        for comp in COMP:
            for lk in LIKES:
                k0, k5, kd = (f'tr|swap|{c}|ep0|S_WT|{comp}|{lk}', f'tr|swap|{c}|ep50|S_WT|{comp}|{lk}',
                              f'tr|swap|{c}|ep50-ep0|S_WT|{comp}|{lk}')
                if k0 in SW:
                    L.append(f'| {CN[c]} | {CMN[comp]} | {LN[lk]} | {fmt(SW[k0])} | ' +
                             (f'{fmt(SW[k5])} | {fmt(SW[kd])} | {cls(SW[k0])}／{cls(SW[k5])}／{cls(SW[kd])} |' if k5 in SW
                              else f'— | — | {cls(SW[k0])} |'))
    L += ['', '## 参考：検査用データ（300）・入れ替えあり、全部', '',
          '| 条件 | 成分 | 0 エポック目 | 50 エポック目 | 50 − 0 |', '|---|---|---|---|---|']
    for c in conds:
        for comp in COMP[:5]:
            k0, k5, kd = (f'te|swap|{c}|ep0|S_WT|{comp}|all', f'te|swap|{c}|ep50|S_WT|{comp}|all',
                          f'te|swap|{c}|ep50-ep0|S_WT|{comp}|all')
            if k0 in SW:
                L.append(f'| {CN[c]} | {CMN[comp]} | {fmt(SW[k0])} | ' + (f'{fmt(SW[k5])} | {fmt(SW[kd])} |' if k5 in SW else '— | — |'))
    open(os.path.join(AN, 's2_tables.md'), 'w').write('\n'.join(L) + '\n')
    fig_s2(runs, curves, mon, pairs)
    print('\n'.join(L[:40]))


def fig_s2(runs, curves, mon, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'p50': 'tab:red', 'sr50': 'tab:blue', 'none': 'tab:gray'}
    ls = {0: '-', 1: '--', 2: ':'}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for j, lk in enumerate(LIKES):
        ax = axes[0, j]
        for (c, s) in sorted(runs):
            cv = curves[f'{c}|{s}|{lk}']
            e = sorted(cv)
            ax.plot(e, [cv[x]['mean'] for x in e], ls[s], marker='.', color=col[c], label=f'{c} s{s}')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(f'D_T(e), {lk}', fontsize=9)
        ax.set_xlabel('epoch')
    axes[0, 0].legend(fontsize=7, ncol=3)
    for j, (q, ttl) in enumerate((('c_self', 'C_self (round 21, hold)'), ('l1_T_V', 'L1 of A-1 view, test, T, flow V'),
                                  ('l1_T_B', 'L1 of A-1 view, test, T, flow B'))):
        ax = axes[1, j]
        for (c, s) in sorted(runs):
            mm = mon[f'{c}|{s}']
            e = sorted(int(x) for x in mm)
            ax.plot(e, [mm[str(x)][q] for x in e], ls[s], marker='.', color=col[c])
        if q == 'c_self':
            ax.axhline(0, color='k', lw=0.6)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('epoch')
    fig.suptitle('round 27 Sec.2: T in place of G, flow B, adapter + shared LSTM (1/10), half stage-I data; solid s0, dashed s1, dotted s2')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(AN, f'fig_round27_s2.{ext}'), dpi=110)


if __name__ == '__main__':
    {'sig': sig, 's2': s2}[sys.argv[1]]()
