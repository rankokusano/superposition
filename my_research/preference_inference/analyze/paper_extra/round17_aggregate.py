"""
Paper round 17 Sec.2.4 aggregation (docs/paper_extra_log.md Sec.41; exploratory, outside the pre-registration). CPU,
saved Round17/eval and Round16/eval arrays only. Terms tot = L1 + FP1 + FP2 (weights 1; FP1 does not depend on Q2 and
cancels in every difference), L1, FP2. Per point (m = mean over the episodes used, steps 0..49):
  D_G = m(z) - m(a)  (round-16 values);  for each set F:
  R2_F = 1 - sum ss_res_F / sum ss_tot  (per-episode sums, steps 0..49, about the checkpoint's per-dimension mean);
  S_F = [m(z) - m(fit_F)]/D_G;  Sres_F = [m(z) - m(res_F)]/D_G;  Delta_F = [m(res_F) - m(a)]/D_G;
  increments S_F1 - S_F0, S_F2 - S_F1, S_F3 - S_F2, S_F4 - S_F3, S_F5 - S_F4 (and the same for Sres).
  Pair value = mean over the 5 late points.
Conditions: p50 (v12_s2_pref_mixed, G ep160-200), none (v12_s2_pref_random); difference p50 - none (same pair);
one-bit G (v14_s2u_pref_mixed[80], 3 pairs, F0 / F4 / F5, descriptive).
CI: t 95% over pairs (df = pairs - 1); two-stage bootstrap 1000 (default_rng(0); each replicate draws the pairs with
replacement and ONE resample of the 1200 episodes shared by every pair, condition and checkpoint), 2.5 / 97.5
percentiles (round16_aggregate.tci / pct).
-> data/result/paper_extra/Round17/analysis/{summary.json, tables.md, fig_round17_S.{pdf,png}}
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round16_aggregate as A16  # noqa  (tci, pct)

RES = A16.RES
EV16 = os.path.join(RES, 'paper_extra', 'Round16', 'eval')
EV = os.path.join(RES, 'paper_extra', 'Round17', 'eval')
OUT = os.path.join(RES, 'paper_extra', 'Round17', 'analysis')
EPS = A16.EPS
N, B = 1200, 1000
TERMS = ('tot', 'L1', 'FP2')
SETS = ('F0', 'F1', 'F2', 'F3', 'F4', 'F5')
MAIN = {'p50': 'v12_s2_pref_mixed', 'none': 'v12_s2_pref_random'}
ONEBIT = {'onebit50': 'v14_s2u_pref_mixed', 'onebit80': 'v14_s2u_pref_mixed80'}


def load(exp, s, e, sets):
    z16 = np.load(os.path.join(EV16, f'{exp}_s{s}_ep{e}.npz'))
    z = np.load(os.path.join(EV, f'{exp}_s{s}_ep{e}.npz'))
    keys, rows = [], []

    def add(k, parts):
        for t, a in zip(TERMS, [sum(parts), parts[0], parts[2]]):
            keys.append(f'{k}|{t}'); rows.append(a)
    for v in ('a', 'z'):
        add(v, [z16[f'{v}|0_49|{t}'].astype(np.float64) for t in ('L1', 'FP1', 'FP2')])
    for f in sets:
        for kind in ('fit', 'res'):
            add(f'{f}|{kind}', [z[f'{f}|{kind}|0_49|{t}'].astype(np.float64) for t in ('L1', 'FP1', 'FP2')])
        keys.append(f'ssres|{f}'); rows.append(z[f'ss_res|{f}'])
    keys.append('sstot'); rows.append(z['ss_tot'])
    return keys, np.stack(rows)


def quantities(m, M, sets):
    """m: episode means, M: episode sums (for R^2)."""
    q = {}
    for f in sets:
        q[f'R2|{f}'] = 1 - M[f'ssres|{f}'] / M['sstot']
    for t in TERMS:
        D = m[f'z|{t}'] - m[f'a|{t}']
        q[f'{t}|D_G'] = D
        for f in sets:
            q[f'{t}|S|{f}'] = (m[f'z|{t}'] - m[f'{f}|fit|{t}']) / D
            q[f'{t}|Sres|{f}'] = (m[f'z|{t}'] - m[f'{f}|res|{t}']) / D
            q[f'{t}|Delta|{f}'] = (m[f'{f}|res|{t}'] - m[f'a|{t}']) / D
        for a, b in zip(sets[:-1], sets[1:]):
            q[f'{t}|dS|{b}-{a}'] = q[f'{t}|S|{b}'] - q[f'{t}|S|{a}']
            q[f'{t}|dSres|{b}-{a}'] = q[f'{t}|Sres|{b}'] - q[f'{t}|Sres|{a}']
    return q


def pair_values(data, idx, sets):
    out = {}
    for pts in data:
        vals = []
        for k, X in pts:
            sub = X[:, idx]
            vals.append(quantities(dict(zip(k, sub.mean(1))), dict(zip(k, sub.sum(1))), sets))
        for name in vals[0]:
            out.setdefault(name, []).append(float(np.mean([v[name] for v in vals])))
    return {k: np.array(v) for k, v in out.items()}


def block(conds, npairs, diffs, sets):
    D = {c: [[load(exp, s, e, sets) for e in EPS] for s in range(npairs)] for c, exp in conds.items()}
    pv = {c: pair_values(D[c], np.arange(N), sets) for c in D}
    rng = np.random.default_rng(0)
    boot = {c: {} for c in list(D) + [f'{a}-{b}' for a, b in diffs]}
    for _ in range(B):
        pi, ei = rng.integers(0, npairs, npairs), rng.integers(0, N, N)
        r = {c: pair_values([D[c][j] for j in pi], ei, sets) for c in D}
        for c in D:
            for k, v in r[c].items():
                boot[c].setdefault(k, []).append(float(v.mean()))
        for a, b in diffs:
            for k in r[a]:
                boot[f'{a}-{b}'].setdefault(k, []).append(float((r[a][k] - r[b][k]).mean()))
    S = {c: {k: dict(A16.tci(v), boot=A16.pct(boot[c][k])) for k, v in pv[c].items()} for c in D}
    for a, b in diffs:
        S[f'{a}-{b}'] = {k: dict(A16.tci(pv[a][k] - pv[b][k]), boot=A16.pct(boot[f'{a}-{b}'][k])) for k in pv[a]}
    return S


def run_info(conds, npairs):
    return {f'{c}|s{s}|ep{e}': json.load(open(os.path.join(EV, f'{exp}_s{s}_ep{e}.json')))
            for c, exp in conds.items() for s in range(npairs) for e in EPS}


def fmt(x, d=3):
    return (f"{x['mean']:+.{d}f} [{x['ci95'][0]:+.{d}f}, {x['ci95'][1]:+.{d}f}]・"
            f"[{x['boot'][0]:+.{d}f}, {x['boot'][1]:+.{d}f}]")


def tables(S, info):
    L = []
    cols = ('p50', 'none', 'p50-none')
    for t in TERMS:
        L += [f'## 項 {t}（対象フレーム 1〜50。組の平均 [t の 95% CI]・[2段のブートストラップ 95%]）', '',
              f"D_G：" + '、'.join(f"{c} {fmt(S['main'][c][f'{t}|D_G'], 4)}" for c in cols), '',
              '| 組 | 量 | ' + ' | '.join(cols) + ' |', '|---|---|' + '---|' * len(cols)]
        for f in SETS:
            for qn in ('S', 'Sres', 'Delta'):
                L.append(f'| {f} | {qn} | ' + ' | '.join(fmt(S['main'][c][f'{t}|{qn}|{f}']) for c in cols) + ' |')
        for a, b in zip(SETS[:-1], SETS[1:]):
            for qn in ('dS', 'dSres'):
                L.append(f'| {b}−{a} | {qn} | ' + ' | '.join(fmt(S['main'][c][f'{t}|{qn}|{b}-{a}']) for c in cols) + ' |')
        L.append('')
    L += ['## 決定係数 R²（Q2、対象フレーム 1〜50、交差当てはめ）', '', '| 組 | ' + ' | '.join(cols) + ' |',
          '|---|' + '---|' * len(cols)]
    for f in SETS:
        L.append(f'| {f} | ' + ' | '.join(fmt(S['main'][c][f'R2|{f}']) for c in cols) + ' |')
    L += ['', '## 1ビットの G（3 組、記述）', '', '| 量 | onebit50 | onebit80 |', '|---|---|---|']
    for t in TERMS:
        L.append(f"| {t} D_G | " + ' | '.join(fmt(S['onebit'][c][f'{t}|D_G'], 4) for c in ONEBIT) + ' |')
        for f in ('F0', 'F4', 'F5'):
            for qn in ('S', 'Sres', 'Delta'):
                L.append(f'| {t} {qn} {f} | ' + ' | '.join(fmt(S['onebit'][c][f'{t}|{qn}|{f}']) for c in ONEBIT) + ' |')
        L.append(f'| {t} dS F5−F4 | ' + ' | '.join(fmt(S['onebit'][c][f'{t}|dS|F5-F4']) for c in ONEBIT) + ' |')
    for f in ('F0', 'F4', 'F5'):
        L.append(f'| R² {f} | ' + ' | '.join(fmt(S['onebit'][c][f'R2|{f}']) for c in ONEBIT) + ' |')
    L += ['', '## 切り詰めの割合（全 100 ステップ × 8 成分、条件の全チェックポイントの平均）と ridge の α', '',
          '| 条件 | ' + ' | '.join(f'{f} fit・res' for f in SETS) + ' |', '|---|' + '---|' * len(SETS)]
    for c in list(MAIN) + list(ONEBIT):
        ks = [k for k in info if k.split('|')[0] == c]
        row = []
        for f in SETS:
            if f'{f}|fit' in info[ks[0]]['clip']:
                row.append('%.2f%%・%.2f%%' % tuple(100 * np.mean([info[k]['clip'][f'{f}|{kd}'] for k in ks]) for kd in ('fit', 'res')))
            else:
                row.append('—')
        L.append(f'| {c} | ' + ' | '.join(row) + ' |')
    L += ['']
    for c in MAIN:
        ks = [k for k in info if k.split('|')[0] == c]
        for f in ('F1', 'F2', 'F3'):
            al = np.concatenate([info[k]['fit'][f]['alpha'] for k in ks])
            L.append(f'- {c} {f} α：' + '、'.join(f'{a:g}×{n}' for a, n in zip(*np.unique(al, return_counts=True))))
    return '\n'.join(L)


def fig(S):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9})
    fg, ax = plt.subplots(2, 2, figsize=(8, 5.4), sharex=True, sharey='row')
    x = np.arange(len(SETS))
    for j, (c, lab) in enumerate((('p50', '50% G'), ('none', 'no-experience G'))):
        for i, (q, yl) in enumerate((('S', 'S_F = [L(z)-L(fit)]/D_G'), ('Sres', 'S_F^res = [L(z)-L(res)]/D_G'))):
            a = ax[i, j]
            v = np.array([S['main'][c][f'tot|{q}|{f}']['per_pair'] for f in SETS])
            for p in range(v.shape[1]):
                a.plot(x, v[:, p], color='0.65', lw=0.8, marker='o', ms=2)
            m = np.array([S['main'][c][f'tot|{q}|{f}']['mean'] for f in SETS])
            ci = np.array([S['main'][c][f'tot|{q}|{f}']['ci95'] for f in SETS])
            a.plot(x, m, color='k', lw=2, marker='o', ms=3)
            a.fill_between(x, ci[:, 0], ci[:, 1], color='k', alpha=0.12, lw=0)
            a.axhline(0, color='k', ls='--', lw=0.7)
            a.axhline(1, color='k', ls=':', lw=0.7)
            a.set_title(lab if i == 0 else '')
            if j == 0:
                a.set_ylabel(yl + '\n(total loss)')
            a.set_xticks(x)
            a.set_xticklabels(SETS)
    fg.tight_layout()
    for ext in ('pdf', 'png'):
        fg.savefig(os.path.join(OUT, f'fig_round17_S.{ext}'), dpi=200)
    plt.close(fg)


def main():
    os.makedirs(OUT, exist_ok=True)
    S = {'note': 'exploratory, outside the pre-registration (Sec.41)',
         'main': block(MAIN, 4, (('p50', 'none'),), SETS),
         'onebit': block(ONEBIT, 3, (), ('F0', 'F4', 'F5'))}
    info = dict(run_info(MAIN, 4), **run_info(ONEBIT, 3))
    S['runs'] = {k: {x: v[x] for x in ('fit', 'clip', 't_fit_s', 't_s')} for k, v in info.items()}
    json.dump(S, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'tables.md'), 'w').write(tables(S, info))
    fig(S)
    print('done', flush=True)


if __name__ == '__main__':
    main()
