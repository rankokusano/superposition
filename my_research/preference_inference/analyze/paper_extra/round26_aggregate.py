"""
Paper round 26 aggregation (docs/paper_extra_log.md Sec.50.2; definitions fixed there before computing).
  sig  Sec.1: S_WT = L(W) - L(T), S_ZT = L(Z) - L(T) (window sums; L = the 3 stage-II loss components summed, and per
       component), relative = S / mean L(T) of the pair and group; groups r25 swap / no swap, r20 O (swap) / S (no swap);
       flows V / B; conditions none / p50 / sr50 / p80 and p50 - sr50, p50 - none; 6-pair t CI + two-stage bootstrap
       (round21_aggregate.stat); reading of the Sec.1 table. -> Round26/analysis/sig_{summary.json,tables.md},
       fig_round26_sig.{pdf,png}
  s2   Sec.2: D_T(e) = C(T) - C(Z), C = Delta(S) - Delta(O); D_T,late = mean of epochs 40-50 per episode; episode
       bootstrap (1 pair) and, if pairs 2-3 ran, 3-pair t CI + two-stage bootstrap; adapter weights; Sec.2 table.
       -> Round26/analysis/s2_{summary.json,tables.md}, fig_round26_s2.{pdf,png}, fig_round26_adapter.{pdf,png}
  s3   Sec.3.1 table (data31/summary.json) and Sec.3.2 curves (C_self, L1, D) mixed vs round-25 stage 2 (not mixed);
       Sec.3 table. -> Round26/analysis/s3_{summary.json,tables.md}, fig_round26_s3.{pdf,png}
Usage: python analyze/paper_extra/round26_aggregate.py sig|s2|s3
"""

import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round21_aggregate as A  # noqa  (stat, excl)
import vdir_stageA as VA  # noqa  (tsum)

_PI = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(_PI, 'data', 'result')
R26 = os.environ.get('ROUND26_DIR', os.path.join(RES, 'paper_extra', 'Round26'))
R25 = os.path.join(RES, 'paper_extra', 'Round25')
AN = os.path.join(R26, 'analysis')
CONDS = ('p50', 'sr50', 'none', 'p80')
CN = {'p50': '経験50%', 'sr50': '符号無作為（経験50%）', 'none': '経験なし', 'p80': '経験80%（記述）',
      'p50-sr50': '経験50% − 符号無作為', 'p50-none': '経験50% − 経験なし'}
COMP = ('total', 'self_vision', 'feat_self', 'feat_other')
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


def fmt(st):
    t, b = st['t']['ci95'], st['boot']
    return f"{st['mean']:+.4f} [{t[0]:+.4f}, {t[1]:+.4f}] [{b[0]:+.4f}, {b[1]:+.4f}]"


# ---------------------------------------------------------------- Sec.1
def sig_arrays():
    Z = {c: [np.load(os.path.join(R26, 'signal', f'{c}_s{s}.npz')) for s in range(6)] for c in CONDS}
    sw = Z['p50'][0]['r25_swap'].astype(bool)
    sel = {'r25_swap': ('r25', sw), 'r25_noswap': ('r25', ~sw), 'r20_O': ('r20O', None), 'r20_S': ('r20S', None)}
    X = {}
    for c in CONDS:
        for g, (pre, m) in sel.items():
            for f in ('V', 'B'):
                L = {}
                for i in ('T', 'W', 'Z'):
                    a = np.stack([z[f'{pre}_{f}_{i}'] for z in Z[c]])          # (6, N, 3)
                    a = a if m is None else a[:, m]
                    L[i] = np.concatenate([a.sum(-1, keepdims=True), a], -1)  # (6, N, 4): total + components
                for q, (a, b) in (('S_WT', ('W', 'T')), ('S_ZT', ('Z', 'T'))):
                    for k, comp in enumerate(COMP):
                        d = L[a][..., k] - L[b][..., k]
                        X[(c, g, f, q, comp)] = d
                        X[(c, g, f, q + '_rel', comp)] = d / L['T'][..., k].mean(1, keepdims=True)
                for k, comp in enumerate(COMP):
                    X[(c, g, f, 'L_T', comp)] = L['T'][..., k]
    return X, Z


def sig():
    os.makedirs(AN, exist_ok=True)
    X, Z = sig_arrays()
    S = {}
    for key, a in X.items():
        S['|'.join(key)] = A.stat(a)
    for c in ('sr50', 'none'):
        for g in GROUPS:
            for f in ('V', 'B'):
                for q in ('S_WT', 'S_WT_rel', 'S_ZT'):
                    for comp in COMP:
                        S['|'.join((f'p50-{c}', g, f, q, comp))] = A.stat(X[('p50', g, f, q, comp)] - X[(c, g, f, q, comp)])
    # reading (main: p50, r25 swap, total S_WT)
    rd = {}
    for c in CONDS:
        for g in GROUPS:
            v, b = cls(S[f'{c}|{g}|V|S_WT|total']), cls(S[f'{c}|{g}|B|S_WT|total'])
            row = {('≈0', '>0'): 1, ('>0', '>0'): 2, ('≈0', '≈0'): 3}.get((v, b))
            rd[f'{c}|{g}'] = {'V': v, 'B': b, 'row': row}
    # swap-unrelated gain: no-swap S_WT > 0 in the same flow and (swap - no swap) per-pair difference, 6-pair t CI has 0
    nos = {}
    for c in CONDS:
        for f in ('V', 'B'):
            ps = np.nanmean(X[(c, 'r25_swap', f, 'S_WT', 'total')], 1)
            pn = np.nanmean(X[(c, 'r25_noswap', f, 'S_WT', 'total')], 1)
            t = VA.tsum((ps - pn).tolist())
            nos[f'{c}|{f}'] = {'swap_minus_noswap': t, 'noswap_cls': cls(S[f'{c}|r25_noswap|{f}|S_WT|total']),
                               'unrelated_to_swap': bool(cls(S[f'{c}|r25_noswap|{f}|S_WT|total']) == '>0' and
                                                         t['ci95'][0] <= 0 <= t['ci95'][1])}
    json.dump({'stats': S, 'reading': rd, 'noswap': nos, 'sec': {f'{c}_s{s}': float(Z[c][s]['sec']) for c in CONDS
                                                                for s in range(6)}},
              open(os.path.join(AN, 'sig_summary.json'), 'w'), indent=1, ensure_ascii=False)
    L = ['# 第26弾 §1 学習の信号の大きさ', '',
         '値：平均 [6 組の t の 95% CI] [2段のブートストラップの 95% CI]。窓（入れ替えから 20 ステップ）の合計。'
         '$L$ = A-1 の視界の L1 + 自己の特徴 + 他者の特徴（段階IIの損失）。', '',
         '## 主：$S_{\\mathrm{WT}} = L(W) - L(T)$（$L$ 全体）', '',
         '| 条件・差 | 群 | V | B | V の判定 | B の判定 | 行 |', '|---|---|---|---|---|---|---|']
    for c in CONDS + ('p50-sr50', 'p50-none'):
        for g in GROUPS:
            sv, sb = S[f'{c}|{g}|V|S_WT|total'], S[f'{c}|{g}|B|S_WT|total']
            row = rd.get(f'{c}|{g}', {}).get('row', '') if c in CONDS else ''
            L.append(f'| {CN[c]} | {GN[g]} | {fmt(sv)} | {fmt(sb)} | {cls(sv)} | {cls(sb)} | {row or "—"} |')
    L += ['', '## 相対の得（$S_{\\mathrm{WT}}$ ÷ $L(T)$ の平均）と、$L(T)$ の大きさ', '',
          '| 条件 | 群 | V 相対 | B 相対 | $L(T)$ V | $L(T)$ B |', '|---|---|---|---|---|---|']
    for c in CONDS:
        for g in GROUPS:
            L.append(f"| {CN[c]} | {GN[g]} | {fmt(S[f'{c}|{g}|V|S_WT_rel|total'])} | {fmt(S[f'{c}|{g}|B|S_WT_rel|total'])} | "
                     f"{S[f'{c}|{g}|V|L_T|total']['mean']:.3f} | {S[f'{c}|{g}|B|L_T|total']['mean']:.3f} |")
    L += ['', '## 成分ごとの $S_{\\mathrm{WT}}$（第25弾・入れ替えあり）', '',
          '| 条件 | 流し方 | A-1 の視界の L1 | 自己の特徴 | 他者の特徴 |', '|---|---|---|---|---|']
    for c in CONDS:
        for f in ('V', 'B'):
            L.append(f'| {CN[c]} | {f} | ' + ' | '.join(fmt(S[f'{c}|r25_swap|{f}|S_WT|{k}']) for k in COMP[1:]) + ' |')
    L += ['', '## 参考：$S_{\\mathrm{ZT}} = L(Z) - L(T)$（$L$ 全体）', '',
          '| 条件 | 群 | V | B |', '|---|---|---|---|']
    for c in CONDS:
        for g in GROUPS:
            L.append(f"| {CN[c]} | {GN[g]} | {fmt(S[f'{c}|{g}|V|S_ZT|total'])} | {fmt(S[f'{c}|{g}|B|S_ZT|total'])} |")
    L += ['', '## 入れ替えと関係のない得か（第25弾、$L$ 全体の $S_{\\mathrm{WT}}$）', '',
          '| 条件 | 流し方 | 入れ替えなしの判定 | あり − なし（組ごとの差、6 組の t CI） | 入れ替えと関係のない得 |', '|---|---|---|---|---|']
    for c in CONDS:
        for f in ('V', 'B'):
            n = nos[f'{c}|{f}']
            t = n['swap_minus_noswap']
            L.append(f"| {CN[c]} | {f} | {n['noswap_cls']} | {t['mean']:+.4f} [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | "
                     f"{'はい' if n['unrelated_to_swap'] else 'いいえ'} |")
    open(os.path.join(AN, 'sig_tables.md'), 'w').write('\n'.join(L) + '\n')
    fig_sig(X)
    print('\n'.join(L[:40]))


def fig_sig(X):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'p50': 'tab:red', 'sr50': 'tab:blue', 'none': 'tab:gray', 'p80': 'tab:orange'}
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    for r, q in enumerate(('S_WT', 'S_WT_rel')):
        for j, g in enumerate(GROUPS):
            ax = axes[r, j]
            for ci, c in enumerate(CONDS):
                for fi, f in enumerate(('V', 'B')):
                    xpos = ci * 3 + fi
                    pm = np.nanmean(X[(c, g, f, q, 'total')], 1)
                    ax.scatter(np.full(6, xpos), pm, s=14, color=col[c], alpha=0.6, marker='o' if f == 'V' else 's')
                    ax.errorbar(xpos, pm.mean(), yerr=1.96 * pm.std(ddof=1) / np.sqrt(6) * 2.571 / 1.96, fmt='_',
                                color='k', ms=14, capsize=3)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xticks([ci * 3 + 0.5 for ci in range(len(CONDS))])
            ax.set_xticklabels([f'{c}\nV | B' for c in CONDS], fontsize=8)
            ax.set_title(f'{g}: {q} (total)', fontsize=9)
    fig.suptitle('round 26 Sec.1: S_WT = L(W) - L(T) per pair (dots) and mean with 6-pair t CI; circles V, squares B')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(AN, f'fig_round26_sig.{ext}'), dpi=110)


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


def s2_runs():
    runs = {}
    for p in sorted(glob.glob(os.path.join(R26, 'runs', 's2_*'))):
        name = os.path.basename(p)
        _, c, f, s = name.split('_')
        ev = {int(os.path.basename(e)[2:5]): np.load(e) for e in sorted(glob.glob(os.path.join(p, 'eval', 'ep*.npz')))}
        runs[(c, f, int(s[1:]))] = {'ev': ev, 'log': json.load(open(os.path.join(p, 'log.json'))),
                                    'done': os.path.exists(os.path.join(p, 'done.json'))}
    return runs


def mv(st):
    return '>0' if st['ci'][0] > 0 else ('<0' if st['ci'][1] < 0 else '≈0')


def s2():
    os.makedirs(AN, exist_ok=True)
    runs = s2_runs()
    S, curves = {}, {}
    for (c, f, s), r in runs.items():
        for gname, lk in (('all', None), ('red', 1), ('green', 0)):
            curves[f'{c}|{f}|{s}|{gname}'] = {e: boot_ep(d_of(z, lk)) for e, z in r['ev'].items()}
            late = np.nanmean(np.stack([d_of(r['ev'][e], lk) for e in (40, 45, 50) if e in r['ev']]), 0)
            S[f'{c}|{f}|{s}|{gname}|late'] = boot_ep(late)
            S[f'{c}|{f}|{s}|{gname}|ep0'] = boot_ep(d_of(r['ev'][0], lk))
        for o in ('sr50', 'none'):
            if c == 'p50' and (o, f, s) in runs:
                lp = np.nanmean(np.stack([d_of(runs[('p50', f, s)]['ev'][e]) for e in (40, 45, 50)]), 0)
                lo = np.nanmean(np.stack([d_of(runs[(o, f, s)]['ev'][e]) for e in (40, 45, 50)]), 0)
                S[f'p50-{o}|{f}|{s}|all|late'] = boot_ep(lp - lo)
    pairs = sorted({s for (_, _, s) in runs})
    multi = {}
    if len(pairs) > 1:
        for c in ('p50', 'sr50', 'none'):
            for f in ('V', 'B'):
                if all((c, f, s) in runs for s in pairs):
                    Xs = np.stack([np.nanmean(np.stack([d_of(runs[(c, f, s)]['ev'][e]) for e in (40, 45, 50)]), 0)
                                   for s in pairs])
                    multi[f'{c}|{f}'] = A.stat(Xs)
    rd = {}
    for c in ('p50', 'sr50', 'none'):
        if all((c, f, 0) in runs for f in ('V', 'B')):
            v, b = mv(S[f'{c}|V|0|all|late']), mv(S[f'{c}|B|0|all|late'])
            rd[c] = {'V': v, 'B': b, 'row': {('≈0', '>0'): 1, ('>0', '>0'): 2, ('≈0', '≈0'): 3}.get((v, b))}
    q = {}
    for (c, f, s), r in runs.items():
        e_last = max(r['ev'])
        q[f'{c}|{f}|{s}'] = {'t_abs_raw': r['ev'][e_last]['t_abs_raw'].mean(0).round(4).tolist(),
                             't_abs_in': r['ev'][e_last]['t_abs_in'].mean(0).round(4).tolist(),
                             'adapter_w': r['ev'][e_last]['adapter_w'].round(3).tolist(),
                             'adapter_b': r['ev'][e_last]['adapter_b'].round(3).tolist(),
                             'readable_min': float(min(min(float(z[k]) for k in z.files if k.endswith('_readable'))
                                                       for z in r['ev'].values())),
                             'epoch_sec_median': float(np.median([x['sec'] for x in r['log']['epochs']])) if r['log']['epochs'] else None,
                             'last_epoch': e_last, 'done': r['done']}
    json.dump({'stats': S, 'curves': {k: {str(e): v for e, v in d.items()} for k, d in curves.items()}, 'multi': multi,
               'reading': rd, 'runs': q}, open(os.path.join(AN, 's2_summary.json'), 'w'), indent=1, ensure_ascii=False)
    L = ['# 第26弾 §2 入口の上限（足した層だけを学ばせる、生成器の代わりに T）', '',
         '値：平均 [エピソードのブートストラップの 95% CI]。$D_T = C_{other}(T) - C_{other}(Z)$、'
         '$D_{T,late}$ = 40・45・50 エポックの平均。', '',
         '| 組 | 条件・差 | 流し方 | $D_T$ 0 エポック目（全部） | $D_{T,late}$ 全部 | 赤好き | 緑好き | 判定 |',
         '|---|---|---|---|---|---|---|---|']
    f3 = lambda st: f"{st['mean']:+.4f} [{st['ci'][0]:+.4f}, {st['ci'][1]:+.4f}]"  # noqa
    for s in pairs:
        for c in ('p50', 'sr50', 'none'):
            for f in ('V', 'B'):
                if (c, f, s) in runs and f'{c}|{f}|{s}|all|late' in S:
                    L.append(f"| {s + 1} | {CN[c]} | {f} | {f3(S[f'{c}|{f}|{s}|all|ep0'])} | {f3(S[f'{c}|{f}|{s}|all|late'])} | "
                             f"{f3(S[f'{c}|{f}|{s}|red|late'])} | {f3(S[f'{c}|{f}|{s}|green|late'])} | {mv(S[f'{c}|{f}|{s}|all|late'])} |")
        for o in ('sr50', 'none'):
            for f in ('V', 'B'):
                k = f'p50-{o}|{f}|{s}|all|late'
                if k in S:
                    L.append(f"| {s + 1} | {CN['p50-' + o]} | {f} | — | {f3(S[k])} | — | — | {mv(S[k])} |")
    if multi:
        L += ['', f'## 組 {", ".join(str(p + 1) for p in pairs)} をまとめた $D_{{T,late}}$（{len(pairs)} 組の t CI、2段のブートストラップ）', '',
              '| 条件 | 流し方 | 値 | 判定 |', '|---|---|---|---|']
        for k, st in multi.items():
            c, f = k.split('|')
            L.append(f'| {CN[c]} | {f} | {fmt(st)} | {cls(st)} |')
    L += ['', '## 読み方の表の当てはめ（組1、主は経験50%）', '', '```', json.dumps(rd, ensure_ascii=False, indent=1), '```',
          '', '## $D_T(e)$（全部、すべての組。s0 = 組1）', '']
    eps = sorted({e for r in runs.values() for e in r['ev']})
    L += ['| 走らせたもの | ' + ' | '.join(f'ep{e}' for e in eps) + ' |', '|---|' + '---|' * len(eps)]
    for (c, f, s) in sorted(runs):
        cv = curves[f'{c}|{f}|{s}|all']
        L.append(f'| {c}_{f}_s{s} | ' + ' | '.join(f"{cv[e]['mean']:+.3f}" if e in cv else '' for e in eps) + ' |')
    L += ['', '## 足した層（最後の評価）とステップ 52 の |T|・|M(A(T))|', '', '```', json.dumps(q, ensure_ascii=False, indent=1), '```']
    open(os.path.join(AN, 's2_tables.md'), 'w').write('\n'.join(L) + '\n')
    fig_s2(runs, curves, pairs)
    print('\n'.join(L[:30]))


def fig_s2(runs, curves, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'p50': 'tab:red', 'sr50': 'tab:blue', 'none': 'tab:gray'}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharey='row')
    for r, g in enumerate(('all', 'red', 'green')):
        for j, f in enumerate(('V', 'B')):
            ax = axes[r, j]
            for c in ('p50', 'sr50', 'none'):
                for s in pairs:
                    k = f'{c}|{f}|{s}|{g}'
                    if k not in curves:
                        continue
                    e = sorted(curves[k])
                    mu = [curves[k][x]['mean'] for x in e]
                    lo = [curves[k][x]['ci'][0] for x in e]
                    hi = [curves[k][x]['ci'][1] for x in e]
                    ax.plot(e, mu, '-o', ms=3, color=col[c], alpha=1.0 if s == 0 else 0.45, label=f'{c} s{s}')
                    if s == 0:
                        ax.fill_between(e, lo, hi, color=col[c], alpha=0.15)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_title(f'flow {f}, D_T ({g})', fontsize=9)
            ax.set_xlabel('epoch')
            if r == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(AN, f'fig_round26_s2.{ext}'), dpi=110)
    fig, axes = plt.subplots(len(pairs), 6, figsize=(17, 3.2 * len(pairs)), squeeze=False)
    for i, s in enumerate(pairs):
        for j, (c, f) in enumerate([(c, f) for c in ('p50', 'sr50', 'none') for f in ('V', 'B')]):
            ax = axes[i, j]
            if (c, f, s) in runs:
                w = runs[(c, f, s)]['ev'][max(runs[(c, f, s)]['ev'])]['adapter_w']
                im = ax.imshow(w, cmap='RdBu_r', vmin=-np.abs(w).max(), vmax=np.abs(w).max())
                plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f'{c} {f} s{s}: A weight (out x in)', fontsize=8)
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(AN, f'fig_round26_adapter.{ext}'), dpi=110)


# ---------------------------------------------------------------- Sec.3
def run_eval(dirpath):
    out = {}
    for e in sorted(glob.glob(os.path.join(dirpath, 'eval', 'ep*.npz'))):
        z = np.load(e)
        ep = int(os.path.basename(e)[2:5])
        d = (z['S_G_delta'] - z['O_G_delta']) - (z['S_Z_delta'] - z['O_Z_delta'])
        out[ep] = {'D': boot_ep(d), 'D_red': boot_ep(np.where(z['a2_liked'] == 1, d, np.nan)),
                   'D_green': boot_ep(np.where(z['a2_liked'] == 0, d, np.nan)),
                   'l1': float(z['self_l1']) if 'self_l1' in z.files else None,
                   'c_self': boot_ep(z['c_self']) if 'c_self' in z.files else None}
    return out


def s3():
    os.makedirs(AN, exist_ok=True)
    d31 = json.load(open(os.path.join(R26, 'data31', 'summary.json')))
    R = {}
    for r in ('0.1', '0.01'):
        R[f'mix_{r}'] = run_eval(os.path.join(R26, 'runs', f's32_p50_r{r}'))
        R[f'nomix_{r}'] = run_eval(os.path.join(R25, 'runs', f'stage2_p50_r{r}'))
    keep = {}
    for k, ev in R.items():
        c0 = ev[0]['c_self']['mean']
        ratios = [ev[e]['c_self']['mean'] / c0 for e in sorted(ev) if e > 0]
        l1r = [ev[e]['l1'] / ev[0]['l1'] for e in sorted(ev) if e > 0]
        keep[k] = {'c_self_start': c0, 'c_self_min_ratio': float(min(ratios)) if ratios else None,
                   'kept_half': bool(ratios and min(ratios) >= 0.5), 'l1_max_ratio': float(max(l1r)) if l1r else None,
                   'last_epoch': max(ev)}
    unl = all(0.45 <= d31[n]['agree_among_moving_with_direction']['value'] <= 0.55 for n in ('s2_pref', 'r25_swap50')) and \
        all('random' in str(d31[n]['attrs'].get('a1_policy', 'random')) or n == 'r25_swap50' for n in ('s2_pref',))
    row = 1 if (unl and keep['mix_0.1']['kept_half']) else (2 if not keep['mix_0.1']['kept_half'] else None)
    json.dump({'data31': d31, 'curves': {k: {str(e): v for e, v in ev.items()} for k, ev in R.items()}, 'keep': keep,
               'unlinked_s2': unl, 'row_main_lr_0.1': row}, open(os.path.join(AN, 's3_summary.json'), 'w'), indent=1,
              ensure_ascii=False)
    f3 = lambda st: f"{st['value']:.3f} [{st['ci'][0]:.3f}, {st['ci'][1]:.3f}]"  # noqa
    L = ['# 第26弾 §3 自己の規則が崩れる理由', '', '## 3.1 データの確かめ（値 [エピソードのブートストラップの 95% CI]）', '',
         '| データ | A-1 の動き（属性） | 動いた歩 | 動いた歩のうち赤に近づいた | 快・不快が 0 でない歩 | 向きが合う（全部） | 規則の歩 | 無作為の歩 | 規則の歩の割合 |',
         '|---|---|---|---|---|---|---|---|---|']
    for n, d in d31.items():
        ar = d['rule_step_share']
        L.append(f"| {n} | {d['attrs'].get('a1_policy', '—')}（p_rule {d['attrs'].get('p_rule', '—')}） | {f3(d['moving_share'])} | "
                 f"{f3(d['closer_to_red_among_moving'])} | {f3(d['pleasure_nonzero_share'])} | {f3(d['agree_among_moving_with_direction'])} | "
                 f"{f3(d['agree_rule_steps']) if d['agree_rule_steps']['den'] else '—'} | {f3(d['agree_random_steps'])} | {ar['value']:.3f} |")
    L += ['', f'- 段階IIのデータで「結びついていない」（当て方は §50.2）：{"はい" if unl else "いいえ"}', '',
          '## 3.2 段階Iのデータを混ぜる（経験50%、組1）', '',
          '| 走らせたもの | エポック | L1 | $C_{self}$ | $D$（全部） |', '|---|---|---|---|---|']
    for k, ev in R.items():
        for e in sorted(ev):
            r = ev[e]
            L.append(f"| {k} | {e} | {r['l1']:.5f} | {r['c_self']['mean']:+.4f} [{r['c_self']['ci'][0]:+.4f}, {r['c_self']['ci'][1]:+.4f}] | "
                     f"{r['D']['mean']:+.4f} [{r['D']['ci'][0]:+.4f}, {r['D']['ci'][1]:+.4f}] |")
    L += ['', '## 半分を保つか（5〜50 エポックのすべての評価で $C_{self}$ ≥ 0.5 × 0 エポック目）', '', '```',
          json.dumps(keep, ensure_ascii=False, indent=1), '```', '', f'- 主（1/10）で当たる行：{row}']
    open(os.path.join(AN, 's3_tables.md'), 'w').write('\n'.join(L) + '\n')
    fig_s3(R)
    print('\n'.join(L))


def fig_s3(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    sty = {'mix_0.1': ('tab:red', '-'), 'mix_0.01': ('tab:red', ':'), 'nomix_0.1': ('tab:gray', '-'), 'nomix_0.01': ('tab:gray', ':')}
    for k, ev in R.items():
        e = sorted(ev)
        c, ls = sty[k]
        axes[0].plot(e, [ev[x]['c_self']['mean'] for x in e], ls, marker='o', ms=3, color=c, label=k)
        axes[1].plot(e, [ev[x]['l1'] for x in e], ls, marker='o', ms=3, color=c, label=k)
        axes[2].plot(e, [ev[x]['D']['mean'] for x in e], ls, marker='o', ms=3, color=c, label=k)
    for ax, t in zip(axes, ('C_self (round 21, hold)', 'L1 of the predicted A-1 view (test)', 'D = C_other(G) - C_other(Z)')):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel('stage-2 epoch')
        ax.axhline(0, color='k', lw=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(AN, f'fig_round26_s3.{ext}'), dpi=110)


if __name__ == '__main__':
    {'sig': sig, 's2': s2, 's3': s3}[sys.argv[1]]()
