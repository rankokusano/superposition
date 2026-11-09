"""
Paper round 30 aggregation (docs/paper_extra_log.md Sec.54.2; quantities and reading rules as pre-registered there).
CPU only. Reads Round30/{s1,s2,s3}/ and writes Round30/analysis/{summary.json, tables.md, fig_round30_*.{png,pdf}}.
Usage: python analyze/paper_extra/round30_aggregate.py
"""

import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vdir_stageA as VA  # noqa
import matplotlib  # noqa
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa

_PI = os.path.dirname(os.path.dirname(_HERE))
R30 = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round30')
OUT = os.path.join(R30, 'analysis')
PAIRS = tuple(range(6))
CONDS = ('p50', 'none', 'sr50')
CN = {'p50': '経験50%', 'none': '経験なし', 'sr50': '符号無作為（経験50%）'}
BS = ['c0167', 'all'] + [f'b{b}' for b in range(8)]
SC = {'a': '(a) 実際の特徴から', 'b': '(b) 予測した特徴から（見せる）', 'c': '(c) 入れ替えの後、止めた予測から', 'ca': '(c の実際) 同じフレームの実際の特徴'}


def r_r2(st):
    n, sx, sy, sxx, syy, sxy, sse = st
    if n < 2:
        return np.nan, np.nan
    vx, vy = n * sxx - sx * sx, n * syy - sy * sy
    r = (n * sxy - sx * sy) / np.sqrt(vx * vy) if vx > 1e-12 and vy > 1e-12 else np.nan
    sst = syy - sy * sy / n
    return r, (1 - sse / sst) if sst > 1e-12 else np.nan


def tsum(v):
    v = [x for x in v if np.isfinite(x)]
    if len(v) >= 2 and np.std(v) == 0:
        return {'mean': float(v[0]), 'ci95': [float(v[0]), float(v[0])], 'n': len(v)}
    if len(v) < 2:
        return {'mean': float(v[0]) if v else None, 'ci95': None, 'n': len(v)}
    return VA.tsum(v)


def stat(Ps):
    """list over pairs of (N, 7) per-episode sufficient statistics -> {'r', 'R2'} with pair t CI and two-stage boot."""
    vals = np.array([r_r2(P.sum(0)) for P in Ps])
    rng = np.random.default_rng(0)
    npr, N = len(Ps), Ps[0].shape[0]
    bs = np.empty((1000, 2))
    for i in range(1000):
        tot = np.zeros(7)
        for p in (rng.integers(0, npr, npr) if npr > 1 else [0]):
            tot += Ps[p][rng.integers(0, N, N)].sum(0)
        bs[i] = r_r2(tot)
    out = {}
    for j, nm in enumerate(('r', 'R2')):
        v = vals[:, j]
        ok = np.isfinite(bs[:, j])
        out[nm] = {'mean': float(np.nanmean(v)) if np.isfinite(v).any() else float('nan'), 'per_pair': v.tolist(),
                   't': tsum(v.tolist()),
                   'boot': [float(np.percentile(bs[ok, j], 2.5)), float(np.percentile(bs[ok, j], 97.5))] if ok.any() else None}
    return out


def fm(st, d=3):
    if not np.isfinite(st['mean']):
        return '—'
    t, b = st['t']['ci95'], st['boot']
    ts = f'[{t[0]:+.{d}f}, {t[1]:+.{d}f}]' if t else '[—]'
    bs = f'[{b[0]:+.{d}f}, {b[1]:+.{d}f}]' if b else '[—]'
    return f"{st['mean']:+.{d}f} {ts} {bs}"


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=110)
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    S, L = {}, ['# 第30弾 他者の快・不快を画像の空間で作れるか（集計）', '',
                '値：平均 [組の t の 95% CI] [2段のブートストラップの 95% CI]（組によらない行はエピソードのブートストラップだけ）。相関は検査のエピソードの全時点をまとめたもの。', '']
    # ---------------- Sec.1
    z1 = [np.load(os.path.join(R30, 's1', f's1_s{s}.npz')) for s in PAIRS]
    zl, zh = np.load(os.path.join(R30, 's1', 'lin.npz')), np.load(os.path.join(R30, 's1', 'hand.npz'))
    for side in ('own', 'other'):
        for bn in BS:
            S[f's1|cnn|{side}|{bn}'] = stat([z[f'{side}|{bn}'] for z in z1])
            S[f's1|lin|{side}|{bn}'] = stat([zl[f'{side}|{bn}']])
            S[f's1|hand|{side}|{bn}'] = stat([zh[f'{side}|{bn}']])
    r1 = S['s1|cnn|other|c0167']['r']['mean']
    read1 = {'cnn_other_r_c0167': r1, 'row': 1 if r1 >= 0.7 else (2 if r1 >= 0.4 else 3)}
    L += ['## §1 自分の経験だけで学んだ網', '', '```', json.dumps(read1, ensure_ascii=False), '```', '',
          '| 読み出し | 当てはめた先 | 範囲 0・1・6・7 の r | $R^2$ | 全範囲の r | $R^2$ |', '|---|---|---|---|---|---|']
    NM = {'cnn': '網（6 種）', 'hand': '手作りの上限', 'lin': '第29弾の線形（Encoder-1）'}
    for k in ('cnn', 'hand', 'lin'):
        for side, sn in (('own', '(a) 自分の視界'), ('other', '(b) 他者の実際の視界')):
            a, b = S[f's1|{k}|{side}|c0167'], S[f's1|{k}|{side}|all']
            L.append(f"| {NM[k]} | {sn} | {fm(a['r'])} | {fm(a['R2'])} | {fm(b['r'])} | {fm(b['R2'])} |")
    L += ['', '範囲ごとの r（網、6 種の平均）：', '', '| 当てはめた先 | ' + ' | '.join(f'b{b}' for b in range(8)) + ' |', '|---|' + '---|' * 8]
    for side in ('own', 'other'):
        L.append(f'| {side} | ' + ' | '.join(('—' if not np.isfinite(S[f's1|cnn|{side}|b{b}']['r']['mean']) else f"{S[f's1|cnn|{side}|b{b}']['r']['mean']:+.3f}") for b in range(8)) + ' |')
    run1 = json.load(open(os.path.join(R30, 's1', 's1_run.json')))
    last = ', '.join('%.2e' % v['train_mse'][-1] for v in run1.values())
    L += ['', f'- 学習の最後のエポックの平均二乗誤差：{last}。', '']
    # ---------------- Sec.2
    z2 = {c: [np.load(os.path.join(R30, 's2', f'{c}_s{s}.npz')) for s in PAIRS] for c in CONDS}
    for c in CONDS:
        for dn in ('exist', 'new'):
            for sc in ('a', 'b', 'c', 'ca'):
                for rd in ('cnn', 'hand') + (('cnn_old',) if sc in ('c', 'ca') else ()):
                    for bn in ('c0167', 'all'):
                        S[f's2|{c}|{dn}|{sc}|{rd}|{bn}'] = stat([z[f'{dn}|{sc}|{rd}|{bn}'] for z in z2[c]])
                pix = np.stack([z[f'{dn}|{sc}|pix_l1'] for z in z2[c]])
                S[f's2|{c}|{dn}|{sc}|pix'] = {'mean': float(pix.mean()), 'per_pair': pix.mean(1).tolist(), 't': tsum(pix.mean(1).tolist())}

    def row2(dn):
        v = {sc: S[f's2|p50|{dn}|{sc}|cnn|c0167']['r']['mean'] for sc in ('a', 'b', 'c')}
        hi = {k: (np.isfinite(x) and x >= 0.7) for k, x in v.items()}
        if not hi['a']:
            r = 4
        elif not hi['b']:
            r = 3
        elif not hi['c']:
            r = 2
        else:
            r = 1
        return {'r': v, 'row': r}
    read2 = {'main_existing_decoder': row2('exist'), 'new_decoder_descriptive': row2('new')}
    L += ['## §2 予測した他者の視点を画像に戻す', '', '```', json.dumps(read2, ensure_ascii=False), '```', '']
    for dn, dname in (('exist', '既存の Visual Decoder（主）'), ('new', '新しく学んだ復号器（記述）')):
        L += [f'### {dname}', '', '| 条件 | 場面 | 網の r（範囲 0・1・6・7） | 手作りの r | 網の r（全範囲） | 画素の誤差 | 入れ替えの前の配置との r（網） |',
              '|---|---|---|---|---|---|---|']
        for c in CONDS:
            for sc in ('a', 'b', 'c', 'ca'):
                old = fm(S[f's2|{c}|{dn}|{sc}|cnn_old|c0167']['r']) if sc in ('c', 'ca') else '—'
                L.append(f"| {CN[c]} | {SC[sc]} | {fm(S[f's2|{c}|{dn}|{sc}|cnn|c0167']['r'])} | {fm(S[f's2|{c}|{dn}|{sc}|hand|c0167']['r'])} | "
                         f"{fm(S[f's2|{c}|{dn}|{sc}|cnn|all']['r'])} | {S[f's2|{c}|{dn}|{sc}|pix']['mean']:.4f} | {old} |")
        L.append('')
    # ---------------- Sec.3
    parts = [np.load(f) for f in sorted(glob.glob(os.path.join(R30, 's3', 'part*of8.npz')))]
    G = {}
    for p in parts:
        for k in p.files:
            G[k] = p[k]
    for nm in ('real_b', 'real_c'):
        S[f's3|{nm}'] = stat([G[f'{nm}|c0167']])
    for sc in ('b', 'c'):
        S[f's3|pred_{sc}'] = stat([G[f'pred_{sc}_s{s}|c0167'] for s in PAIRS])
    found = {k.split('|')[0]: {'found_a2': float(np.mean(G[f"{k.split('|')[0]}|found_a2"])),
                               'found_layout': float(np.mean(G[f"{k.split('|')[0]}|found_layout"])),
                               'layout_correct': float(np.nanmean(G[f"{k.split('|')[0]}|layout_correct"]))}
             for k in G if k.endswith('|found_a2')}
    read3 = {}
    for sc in ('b', 'c'):
        ra, rp = S[f's3|real_{sc}']['r']['mean'], S[f's3|pred_{sc}']['r']['mean']
        read3[sc] = {'real_r': ra, 'pred_r': rp, 'row': 1 if rp >= 0.7 else (2 if ra >= 0.7 else 3)}
    read3['rerender_mismatch'] = {k: int(G[k]) for k in G if k.startswith('rerender')}
    L += ['## §3 予備：幾何', '', '```', json.dumps(read3, ensure_ascii=False), '```', '',
          '| 画像 | 範囲 0・1・6・7 の r | $R^2$ | A-2 が見つかった割合 | 配置が見つかった割合 | 配置の正しさ |', '|---|---|---|---|---|---|']
    for nm, lab in (('real_b', '(b) の実際の画像'), ('pred_b', '(b) の予測画像（経験50%、6 組）'), ('real_c', '(c) の実際の画像'), ('pred_c', '(c) の予測画像（経験50%、6 組）')):
        f = found.get(nm) or {k: float(np.mean([found[f'{nm}_s{s}'][k] for s in PAIRS])) for k in ('found_a2', 'found_layout', 'layout_correct')}
        L.append(f"| {lab} | {fm(S[f's3|{nm}']['r'])} | {fm(S[f's3|{nm}']['R2'])} | {f['found_a2']:.3f} | {f['found_layout']:.3f} | {f['layout_correct']:.3f} |")
    out = {'reading': {'s1': read1, 's2': read2, 's3': read3}, 'stats': S, 'found': found}
    json.dump(out, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, ensure_ascii=False, default=float)
    open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(L) + '\n')
    # ---------------- figures
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    ax = axs[0]
    for j, (k, side) in enumerate([(k, s) for k in ('cnn', 'hand', 'lin') for s in ('own', 'other')]):
        st = S[f's1|{k}|{side}|c0167']['r']
        ax.scatter([j] * len(st['per_pair']), st['per_pair'], s=10, color='C3' if side == 'other' else 'C0')
    ax.set_xticks(range(6))
    ax.set_xticklabels(['CNN own', 'CNN other', 'hand own', 'hand other', 'lin own', 'lin other'], fontsize=7)
    ax.axhline(0.7, color='g', ls='--', lw=0.6)
    ax.axhline(0.4, color='gray', ls=':', lw=0.6)
    ax.set_title('Sec.1 r (ranges 0,1,6,7)', fontsize=9)
    for ai, dn in enumerate(('exist', 'new')):
        ax = axs[1 + ai]
        for ci, c in enumerate(CONDS):
            for j, sc in enumerate(('a', 'b', 'c', 'ca')):
                st = S[f's2|{c}|{dn}|{sc}|cnn|c0167']['r']
                x = j + (ci - 1) * 0.22
                ax.scatter([x] * 6, st['per_pair'], s=8, color={'p50': 'C3', 'none': 'C7', 'sr50': 'C0'}[c], label=c if j == 0 else None)
        ax.set_xticks(range(4))
        ax.set_xticklabels(['(a) real', '(b) pred shown', '(c) pred frozen', '(c) real'], fontsize=8)
        ax.axhline(0.7, color='g', ls='--', lw=0.6)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(f'Sec.2 r of CNN on decoded images ({dn} decoder)', fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, 'fig_round30_r')
    # decoded examples (pair 0, p50)
    import h5py
    z = z2['p50'][0]
    gb = h5py.File(os.path.join(_PI, 'data', 'data', 'v11_s2_pref', 'data.h5'), 'r')['test']['other_vision']
    gc = h5py.File(os.path.join(_PI, 'data', 'data', 'paper_r21_swap', 'data_swap.h5'), 'r')['test']['other_vision']
    real = {'b': gb[0, [1, 31, 61, 91]], 'c': gc[0, [24, 29, 34, 38]]}
    fig, axs = plt.subplots(6, 4, figsize=(14, 7))
    for i, (dn, sc) in enumerate([('real', 'b'), ('exist', 'b'), ('new', 'b'), ('real', 'c'), ('exist', 'c'), ('new', 'c')]):
        ex = real[sc][None] if dn == 'real' else z[f'{dn}|{sc}|examples']
        for j in range(4):
            axs[i, j].imshow(ex[0, j])
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].set_title(f'{dn} {sc} ep0 #{j}', fontsize=7)
    fig.tight_layout()
    save_fig(fig, 'fig_round30_examples')
    print(json.dumps(out['reading'], ensure_ascii=False))


if __name__ == '__main__':
    main()
