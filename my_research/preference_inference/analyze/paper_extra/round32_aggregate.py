"""
Paper round 32 aggregation (docs/paper_extra_log.md Sec.56.2; quantities and reading rules fixed before computing).
CPU. Reads Round32/{dec,a,grid,flow}/ -> Round32/analysis/{summary.json, tables.md, fig_round32_*.{png,pdf}}.
CIs: (a) episodes bootstrap (1000); (b) placements bootstrap (1000); (c) two-stage bootstrap (value nets 0-5 ->
placements, 1000) with the 6-value-net t CI; (d) 6-pair t CI and two-stage bootstrap (pairs -> episodes, 1000).
Range set 0, 1, 6, 7 is the main one.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round30_aggregate as A30  # noqa  (r_r2, tsum, stat)
import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

_PI = os.path.dirname(os.path.dirname(_HERE))
R32 = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round32')
OUT = os.path.join(R32, 'analysis')
PAIRS = tuple(range(6))
CONDS = ('p50', 'none', 'sr50')
BINS = {'c0167': [0, 1, 6, 7], 'all': list(range(8))}
COMPARE = {'geometry_shown_flow_r30': 0.800, 'geometry_frozen_r30': 0.655, 'linear_enc2_r31': 0.700}


def suff(x, y, bins):
    x, y = np.asarray(x, np.float64)[:, bins], np.asarray(y, np.float64)[:, bins]
    return np.stack([np.full(len(x), len(bins), np.float64), x.sum(1), y.sum(1), (x * x).sum(1), (y * y).sum(1),
                     (x * y).sum(1), ((y - x) ** 2).sum(1)], 1)


def point(Ps):
    return float(np.nanmean([A30.r_r2(P.sum(0))[0] for P in Ps]))


def fmt(st):
    return A30.fm(st['r']) if isinstance(st, dict) and 'r' in st else f'{st:+.3f}' if np.isfinite(st) else 'undefined'


def boot_mean(x, B=1000):
    rng = np.random.default_rng(0)
    N = len(x)
    bs = np.array([x[rng.integers(0, N, N)].mean() for _ in range(B)])
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(open(os.path.join(R32, 'dec', 'select.json')))
    ch = sel['chosen']
    main_new = f'{ch}_s0' if ch else None
    res = {'select': sel, 'chosen': ch, 'compare': COMPARE}

    # ---------------- (a) own views (v11_s1_random test)
    za = np.load(os.path.join(R32, 'a', 'a.npz'))
    decs_a = sorted({k.split('|')[0] for k in za.files}, key=lambda d: (d != 'exist', d))
    ra = {}
    for dn in decs_a:
        ra[dn] = {'pix_l1': float(za[f'{dn}|pix_l1'].mean())}
        for mk in ('col_thr', 'col_cont'):
            for bn in BINS:
                P = za[f'{dn}|{mk}|{bn}']
                ra[dn][f'{mk}|{bn}'] = A30.stat([P]) if (dn in ('exist', main_new) and bn == 'c0167') else point([P])
    res['a'] = ra

    # ---------------- (b), (c) grid
    z = np.load(os.path.join(R32, 'grid', 'grid.npz'))
    ok = ~z['overlap']
    lay = z['layout'][ok]
    decs_g = sorted({k.split('|')[0] for k in z.files if '|' in k}, key=lambda d: (d != 'exist', d))
    rb, rc = {}, {}
    for dn in decs_g:
        boot = dn in ('exist', main_new)
        d21, d22 = z[f'{dn}|d_h2_v1'][ok].astype(np.float64), z[f'{dn}|d_h2_v2'][ok].astype(np.float64)
        rb[dn] = {'d_h2_v2': float(d22.mean()), 'd_h2_v1': float(d21.mean()),
                  'd_h1_v1': float(z[f'{dn}|d_h1_v1'][ok].mean()), 'd_h1_v2': float(z[f'{dn}|d_h1_v2'][ok].mean()),
                  'delta': float((d21 - d22).mean()), 'delta_ci': boot_mean(d21 - d22) if boot else None}
        for mk in ('col_thr', 'col_cont', 'pos_thr', 'pos_cont'):
            for bn, bins in BINS.items():
                P = suff(z[f'{dn}|h2_{mk}'][ok], z[f'v2_{mk}'][ok], bins)
                rb[dn][f'h2v2|{mk}|{bn}'] = A30.stat([P]) if (boot and bn == 'c0167' and mk.startswith('col')) else point([P])
            if mk.startswith('col'):
                for li, ln in enumerate(('original', 'swap_rg')):
                    m = lay == li
                    rb[dn][f'h2v2|{mk}|c0167|{ln}'] = point([suff(z[f'{dn}|h2_{mk}'][ok][m], z[f'v2_{mk}'][ok][m], BINS['c0167'])])
                rb[dn][f'h1v1|{mk}|c0167'] = point([suff(z[f'{dn}|h1_{mk}'][ok], z[f'v1_{mk}'][ok], BINS['c0167'])])
        for bn, bins in BINS.items():
            Ps = [suff(z[f'{dn}|h2_cnn{s}'][ok], z['v2_col_thr'][ok], bins) for s in PAIRS]
            rc[dn] = rc.get(dn, {})
            rc[dn][bn] = A30.stat(Ps) if (boot and bn == 'c0167') else point(Ps)
        for li, ln in enumerate(('original', 'swap_rg')):
            m = lay == li
            rc[dn][f'c0167|{ln}'] = point([suff(z[f'{dn}|h2_cnn{s}'][ok][m], z['v2_col_thr'][ok][m], BINS['c0167']) for s in PAIRS])
    Ps = [suff(z[f'v2_cnn{s}'][ok], z['v2_col_thr'][ok], BINS['c0167']) for s in PAIRS]
    rc['real_v2'] = {'c0167': A30.stat(Ps)}
    res['b'], res['c'] = rb, rc

    # ---------------- (d) flows
    rd = {}
    for cond in CONDS:
        F = [np.load(os.path.join(R32, 'flow', f'{cond}_s{s}.npz')) for s in PAIRS]
        decs_d = sorted({k.split('|')[0] for k in F[0].files if not k.startswith('real')}, key=lambda d: (d != 'exist', d))
        for dn in decs_d:
            for sc in ('a', 'b', 'c', 'ca'):
                key = f'{cond}|{dn}|{sc}'
                rd[key] = {'pix_l1': float(np.mean([f[f'{dn}|{sc}|pix_l1'].mean() for f in F]))}
                for ro in ('cnn', 'hand') + (('cnn_old',) if sc in ('c', 'ca') else ()):
                    rd[key][f'{ro}|c0167'] = A30.stat([f[f'{dn}|{sc}|{ro}|c0167'] for f in F])
                    rd[key][f'{ro}|all'] = point([f[f'{dn}|{sc}|{ro}|all'] for f in F])
    res['d'] = rd

    # ---------------- reading table (Sec.56.2)
    if ch is None:
        res['table'] = {'row': 'no_eligible_recipe'}
    else:
        a_thr, a_cont = ra[main_new]['col_thr|c0167']['r']['mean'], ra[main_new]['col_cont|c0167']['r']['mean']
        a_val = max(x for x in (a_thr, a_cont) if np.isfinite(x)) if np.isfinite([a_thr, a_cont]).any() else float('nan')
        c_val = rc[main_new]['c0167']['r']['mean']
        d_val = rd[f'p50|{main_new}|b']['cnn|c0167']['r']['mean']
        if not (np.isfinite(a_val) and a_val >= 0.9):
            row = 4
        elif c_val >= 0.6 and d_val >= 0.6:
            row = 1
        elif c_val >= 0.6:
            row = 2
        elif c_val < 0.4:
            row = 3
        else:
            row = 5
        res['table'] = {'a': a_val, 'c': c_val, 'd_shown': d_val, 'row': row,
                        'd_frozen': rd[f'p50|{main_new}|c']['cnn|c0167']['r']['mean']}
    json.dump(res, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, default=float)
    tables(res, main_new)
    figs(z, main_new, ch)
    print(json.dumps(res['table']))


def tables(res, mn):
    sel = res['select']
    L = ['# 第32弾 集計の表（`round32_aggregate.py`）', '',
         '## 学び方の候補（自分の視界の検証の部分、v11_s1_random 学習用の回 1000〜1199、3 本の平均）', '',
         f"- 既存の復号器：画素の誤差 {sel['existing_val']['l1']:.4f}、赤 − 緑の相関（閾値）{sel['existing_val']['r_thr_c0167']:.3f}、（連続）{sel['existing_val']['r_cont_c0167']:.3f}。画素の誤差の上限 {1.5 * sel['existing_val']['l1']:.4f}。", '',
         '| 候補 | 画素の誤差 | 相関（閾値、0・1・6・7） | 相関（連続） | 乱数 0／1／2 の相関（閾値） | 誤差の条件 |', '|---|---|---|---|---|---|']
    for r, v in sel['recipes'].items():
        ps = '／'.join(f"{x['r_thr_c0167']:.3f}" for x in v['per_seed'])
        L.append(f"| {r} | {v['l1']:.4f} | {v['r_thr_c0167']:.3f} | {v['r_cont_c0167']:.3f} | {ps} | {'満たす' if v['l1_ok'] else '満たさない'} |")
    L += ['', f"- 選んだもの：**{sel['chosen']}**（決め方の量：{sel['key']}）。主の新しい復号器は {mn}。", '',
          '## (a) 自分の視界の作り直し（v11_s1_random の検査用 300 回）', '',
          '| 復号器 | 画素の誤差 | 閾値 0・1・6・7 | 閾値 全範囲 | 連続 0・1・6・7 | 連続 全範囲 |', '|---|---|---|---|---|---|']
    for dn, v in res['a'].items():
        L.append(f"| {dn} | {v['pix_l1']:.4f} | {fmt(v['col_thr|c0167'])} | {fmt(v['col_thr|all'])} | {fmt(v['col_cont|c0167'])} | {fmt(v['col_cont|all'])} |")
    L += ['', '## (b) 元論文と同じ他者の視界の復号（格子 259,920 配置）', '',
          '| 復号器 | d(v̂²,v²) | d(v̂²,v¹) | 差 [CI] | d(v̂¹,v¹) | 色・閾値 | 色・連続 | 位置・閾値 | 位置・連続 | 色・連続 元／入れ替え | 自分側 色・閾値／連続 |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for dn, v in res['b'].items():
        ci = v['delta_ci']
        cis = f" [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ''
        L.append(f"| {dn} | {v['d_h2_v2']:.4f} | {v['d_h2_v1']:.4f} | {v['delta']:+.4f}{cis} | {v['d_h1_v1']:.4f} | {fmt(v['h2v2|col_thr|c0167'])} | "
                 f"{fmt(v['h2v2|col_cont|c0167'])} | {fmt(v['h2v2|pos_thr|c0167'])} | {fmt(v['h2v2|pos_cont|c0167'])} | "
                 f"{fmt(v['h2v2|col_cont|c0167|original'])}／{fmt(v['h2v2|col_cont|c0167|swap_rg'])} | {fmt(v['h1v1|col_thr|c0167'])}／{fmt(v['h1v1|col_cont|c0167'])} |")
    L += ['', '## (c) 一続きの流れ（格子：戻した画像 → 第30弾の価値の網 → A-2 の本当の赤 − 緑）', '',
          '| 復号器 | 0・1・6・7 [網 6 本の t CI] [2段] | 全範囲 | 元の配置だけ | 入れ替えた配置だけ |', '|---|---|---|---|---|']
    for dn, v in res['c'].items():
        L.append(f"| {dn} | {fmt(v['c0167'])} | {fmt(v.get('all', float('nan')))} | {fmt(v.get('c0167|original', float('nan')))} | {fmt(v.get('c0167|swap_rg', float('nan')))} |")
    L += ['', '## (d) 予測の中で（第30弾 §2 の場面、組1〜6）', '',
          '| 条件 | 復号器 | 場面 | 網 0・1・6・7 [t] [2段] | 網 全範囲 | 閾値の計算 0・1・6・7 | 入れ替えの前の配置との相関（網） | 画素の誤差 |', '|---|---|---|---|---|---|---|---|']
    for k, v in res['d'].items():
        c, dn, sc = k.split('|')
        old = fmt(v['cnn_old|c0167']) if 'cnn_old|c0167' in v else '—'
        L.append(f"| {c} | {dn} | {sc} | {fmt(v['cnn|c0167'])} | {fmt(v['cnn|all'])} | {fmt(v['hand|c0167'])} | {old} | {v['pix_l1']:.4f} |")
    L += ['', '## 比べる値', '', f"- {json.dumps(res['compare'])}", '', '## 読み方の表', '', f"- {json.dumps(res['table'], ensure_ascii=False, default=float)}"]
    open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(L) + '\n')


def figs(z, mn, ch):
    ex = np.load(os.path.join(R32, 'grid', 'examples.npz'))
    rows = []
    for lay in ('original', 'swap_rg'):
        for a1, a2 in (((-5, 5), (5, -5)), ((5, 5), (-5, -5)), ((0, -7), (0, 7)), ((-7, 0), (7, 0))):
            rows.append(f'{lay}|A1{a1}|A2{a2}')
    cols = [('v', 0, 'v¹ (A-1 view)'), ('v', 1, 'v² (A-2 view)'), ('exist', 1, 'existing D: v̂²')]
    if mn:
        cols += [(mn, 0, f'new D ({mn}): v̂¹'), (mn, 1, f'new D ({mn}): v̂²')]
    fig, ax = plt.subplots(len(rows), len(cols), figsize=(3.2 * len(cols), 0.95 * len(rows) + 0.6))
    for i, r in enumerate(rows):
        for j, (dn, k, t) in enumerate(cols):
            ax[i, j].imshow(ex[f'{r}|{dn}'][k], aspect='auto', interpolation='nearest')
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(t, fontsize=9)
        ax[i, 0].set_ylabel(r.replace('|', '\n'), fontsize=6, rotation=0, ha='right', va='center')
    fig.suptitle('Round 32: grid decoding with the existing and the new decoder (array channels as RGB)', fontsize=9)
    fig.tight_layout()
    for e in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_round32_examples.{e}'), dpi=130)
    plt.close(fig)
    if mn:
        f0 = np.load(os.path.join(R32, 'flow', 'p50_s0.npz'))
        fig, ax = plt.subplots(6, 4, figsize=(13, 6.5))
        for bi, sc in enumerate(('b', 'c')):
            for j in range(4):
                for k, (src, t) in enumerate((('real', 'real A-2 view'), ('exist', 'existing D'), (mn, f'new D ({mn})'))):
                    a = ax[3 * bi + k, j]
                    a.imshow(f0[f'{src}|{sc}|examples'][0, j], aspect='auto', interpolation='nearest')
                    a.set_xticks([])
                    a.set_yticks([])
                    if j == 0:
                        a.set_ylabel(f'({sc}) {t}', fontsize=7, rotation=0, ha='right', va='center')
        fig.suptitle('Round 32 (d): p50 pair 1, episode 0; (b) shown flow frames 1/31/61/91, (c) swap-frozen steps 1/6/11/15', fontsize=9)
        fig.tight_layout()
        for e in ('png', 'pdf'):
            fig.savefig(os.path.join(OUT, f'fig_round32_flow_examples.{e}'), dpi=130)
        plt.close(fig)


if __name__ == '__main__':
    main()
