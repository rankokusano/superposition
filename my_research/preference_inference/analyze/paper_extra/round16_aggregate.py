"""
Paper round 16 Sec.2.2 aggregation (docs/paper_extra_log.md Sec.40; exploratory, outside the pre-registration). CPU,
saved Round16/eval/*.npz only. Per checkpoint the episode means (steps 0..49 = target frames 1-50, main; 0..99 for
D_G as reference) of the losses for every Q2 input; terms tot = L1 + FP1 + FP2 (weights 1, as in G's training), L1,
FP1, FP2. Per point (m = mean over the episodes used):
  D_G = m(z) - m(a);  R1(s) = [m(a-s) - m(a+s)]/2, R1/D_G;  R2(s) = [m(p-s) - m(p+s)]/2;  Z(s) = m(z) - m(p+s);
  S_pos/res/ax/orth = [m(z) - m(.)]/D_G;  cut = [m(res) - m(a)]/D_G;  R1t(s) = [m(t-s) - m(t+s)]/2, R1t/D_G (tanh form,
  descriptive); dax / dnorm: mean sigma (Q2'-Q2a).c^ and |Q2'-Q2a| over steps 0..49 for a+-s and t+-s.
  Ratios use the D_G of the same term. Pair value = mean over the 5 late points (pre-training: 1 point).
Conditions: p50 (v12_s2_pref_mixed, G ep160-200), none (v12_s2_pref_random), pre (v12_s2_pref_mixed ep0, main
pre-training G), pre_none (v12_s2_pref_random ep0, descriptive); differences p50 - none, p50 - pre (same pair);
one-bit G (v14_s2u_pref_mixed[80], 3 pairs, descriptive: D_G, R1, R1/D_G).
CI: t 95% over pairs (df = pairs - 1); two-stage bootstrap 1000 (default_rng(0); each replicate draws the pairs with
replacement and ONE resample of the 1200 episodes shared by every pair, condition and checkpoint, as stage C),
2.5 / 97.5 percentiles. Clip fractions, (pos) R^2 and the Q2-identity flags are collected from the per-run json.
-> data/result/paper_extra/Round16/analysis/{summary.json, tables.md, fig_round16_R.{pdf,png}, fig_round16_S.{pdf,png}}
"""

import json
import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(_PI, 'data', 'result')
EV = os.path.join(RES, 'paper_extra', 'Round16', 'eval')
OUT = os.path.join(RES, 'paper_extra', 'Round16', 'analysis')
EPS = (160, 170, 180, 190, 200)
SVALS = (1, 2, 4, 7)
TERMS = ('tot', 'L1', 'FP1', 'FP2')
N, B = 1200, 1000
MAIN = {'p50': ('v12_s2_pref_mixed', EPS), 'none': ('v12_s2_pref_random', EPS), 'pre': ('v12_s2_pref_mixed', (0,)),
        'pre_none': ('v12_s2_pref_random', (0,))}
ONEBIT = {'onebit50': ('v14_s2u_pref_mixed', EPS), 'onebit80': ('v14_s2u_pref_mixed80', EPS)}


def variants(onebit):
    v = ['a', 'z'] + [f'a{g}{s}' for s in SVALS for g in '+-']
    if not onebit:
        v += [f'p{g}{s}' for s in SVALS for g in '+-'] + [f't{g}{s}' for s in SVALS for g in '+-'] + ['pos', 'res', 'ax', 'orth']
    return v


def load(exp, s, e, onebit):
    z = np.load(os.path.join(EV, f'{exp}_s{s}_ep{e}.npz'))
    keys, rows = [], []
    for v in variants(onebit):
        for w in ('0_49', '0_99') if v in ('a', 'z') else ('0_49',):
            parts = [z[f'{v}|{w}|{t}'].astype(np.float64) for t in ('L1', 'FP1', 'FP2')]
            for t, a in zip(TERMS, [sum(parts)] + parts):
                keys.append(f'{v}|{w}|{t}'); rows.append(a)
    for k in z.files:
        if k.startswith(('dax|', 'dnorm|')):
            keys.append(k); rows.append(z[k].astype(np.float64))
    return keys, np.stack(rows)


def quantities(m, onebit):
    q = {}
    for t in TERMS:
        D = m[f'z|0_49|{t}'] - m[f'a|0_49|{t}']
        q[f'{t}|D_G'] = D
        q[f'{t}|D_G_0_99'] = m[f'z|0_99|{t}'] - m[f'a|0_99|{t}']
        q[f'{t}|L_a'] = m[f'a|0_49|{t}']
        for s in SVALS:
            r1 = (m[f'a-{s}|0_49|{t}'] - m[f'a+{s}|0_49|{t}']) / 2
            q[f'{t}|R1|{s}'], q[f'{t}|R1/D_G|{s}'] = r1, r1 / D
            if not onebit:
                q[f'{t}|R2|{s}'] = (m[f'p-{s}|0_49|{t}'] - m[f'p+{s}|0_49|{t}']) / 2
                q[f'{t}|Lz-Lp+|{s}'] = m[f'z|0_49|{t}'] - m[f'p+{s}|0_49|{t}']
                r1t = (m[f't-{s}|0_49|{t}'] - m[f't+{s}|0_49|{t}']) / 2
                q[f'{t}|R1t|{s}'], q[f'{t}|R1t/D_G|{s}'] = r1t, r1t / D
        if not onebit:
            for c in ('pos', 'res', 'ax', 'orth'):
                q[f'{t}|S_{c}'] = (m[f'z|0_49|{t}'] - m[f'{c}|0_49|{t}']) / D
            q[f'{t}|cut_res'] = (m[f'res|0_49|{t}'] - m[f'a|0_49|{t}']) / D
    for k in m:
        if k.startswith(('dax|', 'dnorm|')):
            q[k] = m[k]
    return q


def pair_values(data, idx, onebit):
    """data: [pair][point] (keys, X) -> {quantity: array over pairs}."""
    out = {}
    for pts in data:
        vals = [quantities(dict(zip(k, X[:, idx].mean(1))), onebit) for k, X in pts]
        for name in vals[0]:
            out.setdefault(name, []).append(float(np.mean([v[name] for v in vals])))
    return {k: np.array(v) for k, v in out.items()}


def tci(v):
    v = np.asarray(v, float)
    n = len(v)
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return {'mean': float(v.mean()), 'ci95': [float(v.mean() - h), float(v.mean() + h)], 'per_pair': v.tolist()}


def pct(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def block(conds, npairs, diffs, onebit):
    D = {c: [[load(exp, s, e, onebit) for e in eps] for s in range(npairs)] for c, (exp, eps) in conds.items()}
    allidx = np.arange(N)
    pv = {c: pair_values(D[c], allidx, onebit) for c in D}
    rng = np.random.default_rng(0)
    boot = {c: {} for c in list(D) + [f'{a}-{b}' for a, b in diffs]}
    for _ in range(B):
        pi, ei = rng.integers(0, npairs, npairs), rng.integers(0, N, N)
        r = {c: pair_values([D[c][j] for j in pi], ei, onebit) for c in D}
        for c in D:
            for k, v in r[c].items():
                boot[c].setdefault(k, []).append(float(v.mean()))
        for a, b in diffs:
            for k in r[a]:
                boot[f'{a}-{b}'].setdefault(k, []).append(float((r[a][k] - r[b][k]).mean()))
    S = {}
    for c in D:
        S[c] = {k: dict(tci(v), boot=pct(boot[c][k])) for k, v in pv[c].items()}
    for a, b in diffs:
        S[f'{a}-{b}'] = {k: dict(tci(pv[a][k] - pv[b][k]), boot=pct(boot[f'{a}-{b}'][k])) for k in pv[a]}
    return S


def run_info(conds, npairs):
    out = {}
    for c, (exp, eps) in conds.items():
        for s in range(npairs):
            for e in eps:
                out[f'{c}|s{s}|ep{e}'] = json.load(open(os.path.join(EV, f'{exp}_s{s}_ep{e}.json')))
    return out


def fmt(x):
    return f"{x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]・[{x['boot'][0]:+.4f}, {x['boot'][1]:+.4f}]"


def tables(S, info):
    L = ['## 主の量（合計の損失、対象フレーム 1〜50。組の平均 [t の 95% CI]・[2段のブートストラップ 95%]）', '']
    cols = ('p50', 'none', 'pre', 'p50-none', 'p50-pre', 'pre_none')
    L += ['| 量 | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    names = (['D_G', 'D_G_0_99'] + [f'R1|{s}' for s in SVALS] + [f'R1/D_G|{s}' for s in SVALS] + [f'R2|{s}' for s in SVALS]
             + [f'Lz-Lp+|{s}' for s in SVALS] + ['S_pos', 'S_res', 'S_ax', 'S_orth', 'cut_res']
             + [f'R1t|{s}' for s in SVALS] + [f'R1t/D_G|{s}' for s in SVALS])
    for t in TERMS:
        if t != 'tot':
            L += ['', f'## 項 {t}（同じ形）', '', '| 量 | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
        for n in names:
            L.append(f'| {n} | ' + ' | '.join(fmt(S['main'][c][f'{t}|{n}']) for c in cols) + ' |')
    L += ['', '## 1ビットの G（3組、記述。合計の損失）', '', '| 量 | onebit50 | onebit80 |', '|---|---|---|']
    for n in ['D_G'] + [f'R1|{s}' for s in SVALS] + [f'R1/D_G|{s}' for s in SVALS]:
        L.append(f'| {n} | ' + ' | '.join(fmt(S['onebit'][c][f'tot|{n}']) for c in ('onebit50', 'onebit80')) + ' |')
    L += ['', '## (a±)・(t±) で実際に変わった大きさ（対象フレーム 1〜50 の平均。dax ＝ σ(Q2′−Q2)·ĉ、dnorm ＝ ‖Q2′−Q2‖）', '',
          '| 入力 | ' + ' | '.join(cols[:3]) + ' |', '|---|---|---|---|']
    for v in [f'{g}{h}{s}' for g in 'at' for s in SVALS for h in '+-']:
        for kind in ('dax', 'dnorm'):
            k = f'{kind}|{v}'
            L.append(f'| {kind} {v} | ' + ' | '.join(f"{S['main'][c][k]['mean']:+.4f}" for c in cols[:3]) + ' |')
    return '\n'.join(L)


def fig(S):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 9})
    conds = (('p50', '50% G'), ('none', 'no-experience G'), ('pre', 'pre-training G'))
    fg, ax = plt.subplots(2, 3, figsize=(9, 5.2), sharex=True)
    for j, (c, lab) in enumerate(conds):
        for i, (q, yl) in enumerate((('R1', 'R1(s) (total loss)'), ('R2', 'R2(s) (total loss)'))):
            a = ax[i, j]
            v = np.array([S['main'][c][f'tot|{q}|{s}']['per_pair'] for s in SVALS])   # (s, pairs)
            for p in range(v.shape[1]):
                a.plot(SVALS, v[:, p], color='0.65', lw=0.8)
            m = np.array([S['main'][c][f'tot|{q}|{s}']['mean'] for s in SVALS])
            ci = np.array([S['main'][c][f'tot|{q}|{s}']['ci95'] for s in SVALS])
            a.plot(SVALS, m, color='k', lw=2)
            a.fill_between(SVALS, ci[:, 0], ci[:, 1], color='k', alpha=0.12, lw=0)
            a.axhline(0, color='k', ls='--', lw=0.7)
            a.set_title(lab if i == 0 else '')
            if j == 0:
                a.set_ylabel(yl)
            if i == 1:
                a.set_xlabel('s')
    fg.tight_layout()
    for ext in ('pdf', 'png'):
        fg.savefig(os.path.join(OUT, f'fig_round16_R.{ext}'), dpi=200)
    plt.close(fg)
    names = ('S_pos', 'S_res', 'S_ax', 'S_orth', 'cut_res')
    fg, a = plt.subplots(figsize=(8, 3.6))
    w = 0.26
    for j, (c, lab) in enumerate(conds):
        x = np.arange(len(names)) + (j - 1) * w
        m = [S['main'][c][f'tot|{n}']['mean'] for n in names]
        ci = np.array([S['main'][c][f'tot|{n}']['ci95'] for n in names])
        a.bar(x, m, w, label=lab, color=('0.2', '0.55', '0.85')[j], edgecolor='k', lw=0.5)
        a.errorbar(x, m, yerr=[np.array(m) - ci[:, 0], ci[:, 1] - np.array(m)], fmt='none', ecolor='k', lw=0.8, capsize=2)
        for k, n in enumerate(names):
            a.plot([x[k]] * 4, S['main'][c][f'tot|{n}']['per_pair'], 'o', ms=2.5, color='r')
    a.axhline(0, color='k', lw=0.7)
    a.axhline(1, color='k', ls=':', lw=0.7)
    a.set_xticks(np.arange(len(names)))
    a.set_xticklabels(['S_pos', 'S_res', 'S_ax', 'S_orth', '[L(res)-L(a)]/D_G'])
    a.set_ylabel('share of D_G (total loss)')
    a.legend(frameon=False, fontsize=8)
    fg.tight_layout()
    for ext in ('pdf', 'png'):
        fg.savefig(os.path.join(OUT, f'fig_round16_S.{ext}'), dpi=200)
    plt.close(fg)


def main():
    os.makedirs(OUT, exist_ok=True)
    S = {'note': 'exploratory, outside the pre-registration (Sec.40)',
         'main': block(MAIN, 4, (('p50', 'none'), ('p50', 'pre')), False),
         'onebit': block(ONEBIT, 3, (), True)}
    info = dict(run_info(MAIN, 4), **run_info(ONEBIT, 3))
    S['runs'] = {k: {x: v[x] for x in ('Q2_vs_stageC_identical', 'Q2_vs_stageC_maxabs', 'clip', 'r2_pos', 't_s')}
                 for k, v in info.items()}
    json.dump(S, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    open(os.path.join(OUT, 'tables.md'), 'w').write(tables(S, info))
    fig(S)
    print('done', flush=True)


if __name__ == '__main__':
    main()
