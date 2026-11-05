"""
Paper round 20 follow-up (docs/paper_extra_log.md Sec.44.5; exploratory, outside the pre-registration; written and
committed before computing). CPU, saved Round20/eval arrays only (no training, no new evaluation).
Colour-layout effect, using that A-2's path is the same with and without the swap (O == S, N == R):
    I = D_swap + D_noswap = [surprise(O) - surprise(N)] + [surprise(R) - surprise(S)]
      = [surprise(O) - surprise(S)] + [surprise(R) - surprise(N)]     (same path, swapped vs not swapped colours)
surprise = mean FP over steps 53..79 (as in round20_aggregate). Episode value -> mean over checkpoints -> pair value.
Rows: inputs G / T / Z; conditions T, Z: p50, none, sr50 (pairs 0-5, mse family, ep160-200); G: p50, none (pairs 0-3,
ep160-200), pre, pre_none (ep0); vision V1, V2, V0; p_rule 0.8 / 0.5; all episodes.
Differences (same CIs): p50 - none (G, T, Z), p50 - sr50 (T, Z), p50 - pre (G, descriptive),
  T - Z (same model and episodes; p50, none, sr50; 6 pairs),
  G - Z (pair s = 0..3 by seed: G p50 - Z p50, G none - Z none, G pre - Z p50, G pre_none - Z none; Z mean over
         ep160-200 as in all round-20 tables; supplement: Z ep200 only = the checkpoint G's base carries).
CIs: pair t 95% CI and the two-stage bootstrap (1000, default_rng(0)), round20_aggregate.stat.
Reading (Sec.44.5): both CIs exclude 0 above -> 'I>0'; both include 0 -> 'I≈0'; both exclude 0 below -> 'I<0';
the two CIs disagree -> no reading.
Check: the mean of I equals the round-20 summary's D_swap + D_noswap mean for every input / condition / p / vision.
-> Round20/interaction/{summary.json, tables.md}
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round20_aggregate as A  # noqa

OUT = os.path.join(A.V.RES, 'paper_extra', 'Round20', 'interaction')
VMS = ('V1', 'V2', 'V0')
KEYS = [f'p{p}_{vm}' for p in A.PS for vm in VMS]
G_VS_Z = (('p50', 'p50'), ('none', 'none'), ('pre', 'p50'), ('pre_none', 'none'))
READ = {1: 'I>0', 0: 'I≈0', -1: 'I<0', None: 'CI が食い違う'}


def I_values(fam, cond, s, inp, eps=None):
    """-> {'p<p>_<vm>': (N,) episode I (mean over checkpoints)}"""
    fl = A.files(fam, cond, s) if eps is None else [os.path.join(A.EV, f'{fam}_{cond}_s{s}_ep{e}.npz') for e in eps]
    acc = {}
    for f in fl:
        z = np.load(f)
        for p in A.PS:
            for vm in VMS:
                k = f'p{p}_{vm}_{inp}'
                sp = {v: z[f'FP_{k}_{v}'][:, A.W].astype(np.float64).mean(1) for v in A.VARS}
                acc.setdefault(f'p{p}_{vm}', []).append((sp['O'] - sp['N']) + (sp['R'] - sp['S']))
    return {k: np.mean(v, 0) for k, v in acc.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    X = {}       # (inp, cond) -> {key: (npairs, N)}
    for inp, spec in (('G', A.GG), ('T', A.TZ), ('Z', A.TZ)):
        for c in spec['conds']:
            L = [I_values('G' if inp == 'G' else 'mse', c, s, inp) for s in spec['pairs']]
            X[(inp, c)] = {k: np.stack([d[k] for d in L]) for k in KEYS}
    Z200 = {c: [I_values('mse', c, s, 'Z', eps=(200,)) for s in A.GG['pairs']] for c in ('p50', 'none')}
    Z200 = {c: {k: np.stack([d[k] for d in L]) for k in KEYS} for c, L in Z200.items()}

    S = {}
    for (inp, c), d in X.items():
        for k in KEYS:
            S[f'{inp}|{c}|{k}'] = A.stat(d[k])
    for inp, diffs in (('G', (('p50', 'none'), ('p50', 'pre'))), ('T', A.TZ['diffs']), ('Z', A.TZ['diffs'])):
        for a, b in diffs:
            for k in KEYS:
                S[f'{inp}|{a}-{b}|{k}'] = A.stat(X[(inp, a)][k] - X[(inp, b)][k])
    for c in A.TZ['conds']:
        for k in KEYS:
            S[f'T-Z|{c}|{k}'] = A.stat(X[('T', c)][k] - X[('Z', c)][k])
    for g, zc in G_VS_Z:
        for k in KEYS:
            S[f'G-Z|{g}-{zc}|{k}'] = A.stat(X[('G', g)][k] - X[('Z', zc)][k][:len(A.GG['pairs'])])
            S[f'G-Z200|{g}-{zc}|{k}'] = A.stat(X[('G', g)][k] - Z200[zc][k])
    for k, st in S.items():
        st['reading'] = READ[A.excl(st)]

    # check against the round-20 summary (D_swap + D_noswap means)
    R20 = json.load(open(os.path.join(A.OUT, 'summary.json')))['stats']
    maxdev = 0.0
    for (inp, c) in X:
        for k in KEYS:
            ref = R20[f'{inp}|{c}|{k}_{inp}|D_swap|all']['mean'] + R20[f'{inp}|{c}|{k}_{inp}|D_noswap|all']['mean']
            maxdev = max(maxdev, abs(S[f'{inp}|{c}|{k}']['mean'] - ref))
    assert maxdev < 1e-9, maxdev

    focus = {f'T|{c}|p{p}_{vm}': S[f'T|{c}|p{p}_{vm}']['reading'] for c in A.TZ['conds'] for p in A.PS for vm in VMS}
    json.dump({'stats': S, 'check_max_dev_vs_round20_summary': maxdev, 'focus_T': focus},
              open(os.path.join(OUT, 'summary.json'), 'w'), indent=1, ensure_ascii=False)

    def row(key):
        return ' | '.join(f"{A.fmt(S[f'{key}|p{p}_{vm}'])} **{S[f'{key}|p{p}_{vm}']['reading']}**" for vm in VMS)

    Lt = ['# 第20弾の追加：色の配置の効果 $I = D_{swap} + D_{noswap}$（探索・事前登録外）', '',
          '値：組の値の平均 [組の t の 95% CI] [2段のブートストラップの 95% CI] と読み。G は 4 組、T・Z は 6 組。評価区間はステップ 53〜79。',
          f'確かめ：$I$ の平均と第20弾の $D_{{swap}}$ + $D_{{noswap}}$ の平均の差の最大 {maxdev:.1e}。', '']
    for p in A.PS:
        Lt += [f'## p_rule 0.{p // 10}', '', '| 入力 | 条件・差 | V1 | V2 | V0 |', '|---|---|---|---|---|']
        for inp, conds in (('G', ('p50', 'none', 'pre', 'pre_none', 'p50-none', 'p50-pre')),
                           ('T', ('p50', 'none', 'sr50', 'p50-none', 'p50-sr50')),
                           ('Z', ('p50', 'none', 'sr50', 'p50-none', 'p50-sr50'))):
            for c in conds:
                Lt.append(f'| {inp} | {c} | ' + ' | '.join(
                    f"{A.fmt(S[f'{inp}|{c}|p{p}_{vm}'])} **{S[f'{inp}|{c}|p{p}_{vm}']['reading']}**" for vm in VMS) + ' |')
        for c in A.TZ['conds']:
            Lt.append(f'| T − Z | {c} | ' + ' | '.join(
                f"{A.fmt(S[f'T-Z|{c}|p{p}_{vm}'])} **{S[f'T-Z|{c}|p{p}_{vm}']['reading']}**" for vm in VMS) + ' |')
        for tag, nm in (('G-Z', 'G − Z'), ('G-Z200', 'G − Z（Z は ep200 だけ）')):
            for g, zc in G_VS_Z:
                Lt.append(f'| {nm} | G {g} − Z {zc} | ' + ' | '.join(
                    f"{A.fmt(S[f'{tag}|{g}-{zc}|p{p}_{vm}'])} **{S[f'{tag}|{g}-{zc}|p{p}_{vm}']['reading']}**"
                    for vm in VMS) + ' |')
        Lt.append('')
    Lt += ['## 入力 T の読み', '', '| 条件 | p_rule | V1 | V2 | V0 |', '|---|---|---|---|---|']
    for c in A.TZ['conds']:
        for p in A.PS:
            Lt.append(f'| {c} | 0.{p // 10} | ' + ' | '.join(focus[f'T|{c}|p{p}_{vm}'] for vm in VMS) + ' |')
    open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(Lt) + '\n')
    print('\n'.join(Lt))


if __name__ == '__main__':
    main()
