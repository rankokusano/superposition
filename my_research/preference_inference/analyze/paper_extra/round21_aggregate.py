"""
Paper round 21 Sec.2 aggregation (docs/paper_extra_log.md Sec.45.2; exploratory, outside the pre-registration; written
and committed before any evaluation output was read). CPU. Input: Round21/eval/<cond>_s<s>_ep<e>.npz (round21_eval.py).
Per episode and checkpoint: Delta = s_Lold - s_Lnew (self: d* slopes, main read-out colour-agnostic, supplement '_cur'
= the data's current layout; other: A-2 position slopes). C = Delta(noswap) - Delta(swap) (same episode, same
checkpoint, same flow V1 / V0 and pleasure variant hold / flip / zero). Swap hold - flip = Delta(swap, hold) -
Delta(swap, flip).
Episode value = mean over the checkpoints where it is finite; pair value = mean over finite episodes. CIs: pair t 95%
CI (df 5) and two-stage bootstrap (1000, default_rng(0); pairs resampled, then episodes within each selected pair;
non-finite episodes skipped); differences p50 - none, p50 - sr50 per episode (same episodes).
Reading (Sec.45.2, fixed): main = p50, V1, hold, colour-agnostic self read-out. '>0' / '<0' = both CIs exclude 0 on
that side, '≈0' = both include 0, CIs disagree = no reading.
-> Round21/analysis/{summary.json, tables.md, fig_round21_positions.{pdf,png}, fig_round21_C.{pdf,png}}
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _PI)
import vdir_stageA as VA  # noqa  (tsum)
from sim_v10 import layout as L  # noqa

RES = os.path.join(_PI, 'data', 'result')
EV = os.path.join(RES, 'paper_extra', 'Round21', 'eval')
OUT = os.path.join(RES, 'paper_extra', 'Round21', 'analysis')
DATA = os.path.join(_PI, 'data', 'data', 'paper_r21_swap')
CONDS = ('p50', 'none', 'sr50', 'p80')
CNAME = {'p50': '経験50%', 'none': '経験なし', 'sr50': '符号無作為（経験50%）', 'p80': '経験80%（記述）'}
DIFFS = (('p50', 'none'), ('p50', 'sr50'))
PAIRS = tuple(range(6))
EPS = (160, 170, 180, 190, 200)
VS = ('V1', 'V0')
PV = ('hold', 'flip', 'zero')
PATHS = (('self', ''), ('self', '_cur'), ('other', ''))
B = 1000


def load_ep(c, s):
    """-> dict quantity -> (N,) episode values (mean over finite checkpoints), plus readable fractions and curves."""
    acc, frac, curv = {}, {}, {}
    for e in EPS:
        z = np.load(os.path.join(EV, f'{c}_s{s}_ep{e}.npz'))
        for v in VS:
            for path, tag in PATHS:
                for pv in PV:
                    ds = {d: z[f'{d}_{v}_{path}_{pv}{tag}'].astype(np.float64) for d in ('swap', 'noswap')}
                    for d in ds:
                        acc.setdefault(f'Delta|{d}|{v}|{path}{tag}|{pv}', []).append(ds[d])
                        frac.setdefault(f'read_Delta|{d}|{v}|{path}{tag}|{pv}', []).append(np.isfinite(ds[d]).mean())
                    acc.setdefault(f'C|{v}|{path}{tag}|{pv}', []).append(ds['noswap'] - ds['swap'])
                for d in ('swap', 'noswap'):
                    h, f = z[f'{d}_{v}_{path}_hold{tag}'].astype(np.float64), z[f'{d}_{v}_{path}_flip{tag}'].astype(np.float64)
                    acc.setdefault(f'HmF|{d}|{v}|{path}{tag}', []).append(h - f)
            for d in ('swap', 'noswap'):
                for pv in PV:
                    est = z[f'{d}_{v}_other_{pv}_est']
                    frac.setdefault(f'read_est|{d}|{v}|other|{pv}', []).append(np.isfinite(est[..., 0]).mean())
                    if v == 'V1' and pv == 'hold':
                        curv.setdefault(f'est|{d}', []).append(est.astype(np.float64))
                        curv.setdefault(f'dstar|{d}', []).append(z[f'{d}_{v}_self_{pv}_dstar'].astype(np.float64))
    with np.errstate(all='ignore'):
        ev = {k: np.nanmean(np.stack(v), 0) for k, v in acc.items()}
        cv = {k: np.nanmean(np.stack(v), 0) for k, v in curv.items()}
    return ev, {k: float(np.mean(v)) for k, v in frac.items()}, cv


def boot(X):
    rng = np.random.default_rng(0)
    npairs, N = X.shape
    pi = rng.integers(0, npairs, (B, npairs))
    ei = rng.integers(0, N, (B, npairs, N))
    with np.errstate(all='ignore'):
        m = np.nanmean(np.nanmean(X[pi[:, :, None], ei], 2), 1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def stat(X):
    with np.errstate(all='ignore'):
        pm = np.nanmean(X, 1)
    return {'mean': float(pm.mean()), 't': VA.tsum(pm.tolist()), 'boot': boot(X), 'per_pair': pm.tolist(),
            'n_finite_episodes': int(np.isfinite(X).any(0).sum())}


def excl(st):
    t, b = st['t']['ci95'], st['boot']
    ex_t = 1 if t[0] > 0 else (-1 if t[1] < 0 else 0)
    ex_b = 1 if b[0] > 0 else (-1 if b[1] < 0 else 0)
    return ex_t if ex_t == ex_b else None


SIGN = {1: '>0', 0: '≈0', -1: '<0', None: 'CI が食い違う'}


def fmt(st, d=4):
    t, b = st['t']['ci95'], st['boot']
    return f"{st['mean']:+.{d}f} [{t[0]:+.{d}f}, {t[1]:+.{d}f}] [{b[0]:+.{d}f}, {b[1]:+.{d}f}]"


def reading(S):
    k = lambda q: S[q]  # noqa
    cs, co = excl(k('p50|C|V1|self|hold')), excl(k('p50|C|V1|other|hold'))
    rows = {'row1_self_pos_other_pos': cs == 1 and co == 1,
            'row2_self_pos_other_0': cs == 1 and co == 0,
            'row3_self_0_other_0': cs == 0 and co == 0,
            'row4_self_0_other_pos': cs == 0 and co == 1}
    exp = {f'{a}-{b}|{path}': SIGN[excl(S[f'{a}-{b}|C|V1|{path}|hold'])] for a, b in DIFFS for path in ('self', 'other')}
    checks = {f'swap_hold_minus_flip|{path}': SIGN[excl(S[f'p50|HmF|swap|V1|{path}'])] for path in ('self', 'other')}
    checks.update({f'V0_C|{path}': SIGN[excl(S[f'p50|C|V0|{path}|hold'])] for path in ('self', 'self_cur', 'other')})
    return {'C_self': SIGN[cs], 'C_other': SIGN[co], 'rows': rows, 'no_row': not any(rows.values()),
            'experience_dependence': exp, 'checks': checks}


def main():
    os.makedirs(OUT, exist_ok=True)
    EVV, FR, CU = {}, {}, {}
    for c in CONDS:
        for s in PAIRS:
            EVV[(c, s)], FR[(c, s)], CU[(c, s)] = load_ep(c, s)
    keys = list(EVV[(CONDS[0], 0)].keys())
    S = {}
    for k in keys:
        for c in CONDS:
            S[f'{c}|{k}'] = stat(np.stack([EVV[(c, s)][k] for s in PAIRS]))
        for a, b in DIFFS:
            S[f'{a}-{b}|{k}'] = stat(np.stack([EVV[(a, s)][k] - EVV[(b, s)][k] for s in PAIRS]))
    FRm = {c: {k: float(np.mean([FR[(c, s)][k] for s in PAIRS])) for k in FR[(c, 0)]} for c in CONDS}
    R = reading(S)
    json.dump({'stats': S, 'readable': FRm, 'reading': R}, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1,
              ensure_ascii=False)

    Lt = ['# 第21弾 §2（探索・事前登録外）', '',
          '値：組の値の平均 [組の t の 95% CI] [2段のブートストラップの 95% CI]、6 組。Δ = 古い場所 $L_{old}$ への傾き − 新しい場所 $L_{new}$ への傾き。'
          '$C$ = Δ(入れ替えなし) − Δ(入れ替えあり)。自己の主は色を問わない読み出し、「自己（今の配置）」は補足。', '']
    for v in VS:
        Lt += [f'## $C$（{v}）', '', '| 経路 | 快・不快 | ' + ' | '.join(CNAME[c] for c in CONDS) + ' | 経験50% − 経験なし | 経験50% − 符号無作為 |',
               '|---|---|' + '---|' * (len(CONDS) + 2)]
        for path, tag in PATHS:
            for pv in PV:
                q = f'C|{v}|{path}{tag}|{pv}'
                Lt.append(f'| {path}{tag} | {pv} | ' + ' | '.join(fmt(S[f'{c}|{q}']) for c in CONDS) + ' | ' +
                          ' | '.join(fmt(S[f'{a}-{b}|{q}']) for a, b in DIFFS) + ' |')
        Lt.append('')
    for v in VS:
        Lt += [f'## Δ（{v}）', '', '| 条件 | 経路 | 快・不快 | 入れ替えあり | 入れ替えなし |', '|---|---|---|---|---|']
        for c in CONDS:
            for path, tag in PATHS:
                for pv in PV:
                    Lt.append(f'| {CNAME[c]} | {path}{tag} | {pv} | ' + ' | '.join(
                        fmt(S[f'{c}|Delta|{d}|{v}|{path}{tag}|{pv}']) for d in ('swap', 'noswap')) + ' |')
        Lt.append('')
    Lt += ['## 入れ替えありの「保つ − 反転」の Δ（V1）', '', '| 経路 | ' + ' | '.join(CNAME[c] for c in CONDS) + ' |',
           '|---|' + '---|' * len(CONDS)]
    for path, tag in PATHS:
        Lt.append(f'| {path}{tag} | ' + ' | '.join(fmt(S[f'{c}|HmF|swap|V1|{path}{tag}']) for c in CONDS) + ' |')
    Lt += ['', '## 読めた割合（6 組・5 点の平均）', '', '| 条件 | ' + ' | '.join(
        f'{d} {v} {nm}' for v in VS for d in ('swap', 'noswap') for nm in ('自己 Δ', '他者 Δ', '他者の位置')) + ' |',
        '|---|' + '---|' * 12]
    for c in CONDS:
        Lt.append(f'| {CNAME[c]} | ' + ' | '.join(
            f"{FRm[c][f'read_Delta|{d}|{v}|self|hold']:.3f} | {FRm[c][f'read_Delta|{d}|{v}|other|hold']:.3f} | "
            f"{FRm[c][f'read_est|{d}|{v}|other|hold']:.3f}" for v in VS for d in ('swap', 'noswap')) + ' |')
    Lt += ['', '## 読み方の表（主：経験50%・V1・保つ）', '', '```', json.dumps(R, indent=1, ensure_ascii=False), '```', '']
    open(os.path.join(OUT, 'tables.md'), 'w').write('\n'.join(Lt))
    print('\n'.join(Lt))
    figures(S, CU)


def figures(S, CU):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    g = h5py.File(os.path.join(DATA, 'data_noswap.h5'), 'r')['test']
    lay = g['layout'][()]
    lo = np.array([L.LANDMARKS[L.CODE_LAYOUT[int(c)]]['Red'] for c in lay], np.float64)
    ln = np.array([L.LANDMARKS[L.other_layout(L.CODE_LAYOUT[int(c)])]['Red'] for c in lay], np.float64)
    p23 = g['self_position'][:, 23].astype(np.float64)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for d, ls in (('swap', '-'), ('noswap', ':')):
        ds = np.stack([CU[('p50', s)][f'dstar|{d}'] for s in PAIRS])               # (6, N, 2, K)
        est = np.stack([CU[('p50', s)][f'est|{d}'] for s in PAIRS])                # (6, N, 10, 2)
        for i, (tg, col, nm) in enumerate(((lo, 'C3', 'L_old'), (ln, 'C2', 'L_new'))):
            dist0 = np.linalg.norm(tg - p23, axis=1)                               # (N,)
            with np.errstate(all='ignore'):
                y = np.nanmean(np.nanmean(dist0[None, :, None] - ds[:, :, i], 1), 0)
                yo = np.nanmean(np.nanmean(np.linalg.norm(est - tg[None, :, None], axis=-1), 1), 0)
            axs[0].plot(np.arange(1, len(y) + 1), y, ls, color=col, label=f'{d}: to {nm}')
            axs[1].plot(np.arange(1, len(yo) + 1), yo, ls, color=col, label=f'{d}: to {nm}')
    axs[0].set_title('self (A-1 in the prediction), 50%, V1, hold', fontsize=9)
    axs[1].set_title('other (A-2 in the prediction), 50%, V1, hold', fontsize=9)
    for ax in axs:
        ax.set_xlabel('step k of the freeze (prediction made at t = 23 + k - 1)')
        ax.set_ylabel('distance')
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round21_positions.{ext}'), dpi=130)
    plt.close(fig)
    rows = [(f'{c} {v} {pv}', f'{c}|C|{v}|{{}}|{pv}') for c in CONDS for v in VS for pv in PV] + \
           [(f'{a}-{b} V1 hold', f'{a}-{b}|C|V1|{{}}|hold') for a, b in DIFFS]
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.32 * len(rows) + 1.2), sharey=True)
    for ax, path in zip(axs, ('self', 'other')):
        for i, (nm, q) in enumerate(rows):
            st = S[q.format(path)]
            y = len(rows) - i
            ax.plot(st['per_pair'], [y] * len(st['per_pair']), 'o', color='0.6', ms=3)
            ax.errorbar(st['mean'], y, xerr=[[st['mean'] - st['t']['ci95'][0]], [st['t']['ci95'][1] - st['mean']]],
                        fmt='D', color='C3' if q.startswith('p50|C|V1') and 'hold' in q else 'k', ms=4, capsize=2)
        ax.axvline(0, color='0.3', lw=0.8)
        ax.set_yticks(range(len(rows), 0, -1))
        ax.set_yticklabels([r[0] for r in rows], fontsize=6)
        ax.set_title(f'C_{path} (pairs grey, mean with pair t 95% CI)', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round21_C.{ext}'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
