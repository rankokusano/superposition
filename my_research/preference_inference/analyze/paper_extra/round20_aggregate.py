"""
Paper round 20 Sec.2 aggregation (paper/check/paper_instructions_round20.md; docs/paper_extra_log.md Sec.44.3; exploratory,
outside the pre-registration; written and committed before any evaluation output was read). CPU.
Input: Round20/eval/<fam>_<cond>_s<s>_ep<e>.npz (round20_eval.py).
Steps: step t takes frame t and predicts frame t + 1 (t = 0..79); evaluation window W = steps 53..79 (after the last
shown frame of V1); pre-swap window steps 40..49.
  surprise_e(var) = mean_W FP (other-side feature prediction error)       D_swap = surprise(O) - surprise(N)
                                                                          D_noswap = surprise(R) - surprise(S)
  self error  = mean_W L1;  self-error disturbance = mean(N, O) - mean(S, R)                  (V1, V2)
  tracking    = mean_W tracking error; disturbance = mean(N, O) - mean(S, R); pre-swap error = mean over steps 40..49
                (identical in all variants); flag "tracking disturbed" if disturbance >= 2 x pre-swap error   (V1, V2)
Sources: T / Z: FP and tracking from the mse family, L1 from the l1 family (pairs 0-5, ep160-200);
         G: v12_s2_pref_{mixed,random}_s{0..3} ep160-200 (p50 / none), pre-training ep0 (pre / pre_none, 1 point).
Pair value = mean over its checkpoints of the episode mean. CIs: pair t 95% CI (df = pairs - 1) and the two-stage
bootstrap (1000, default_rng(0): pairs resampled, then episodes within each selected pair; the same episodes for both
members of a difference and for all checkpoints), percentile 2.5 / 97.5.
Rows: experience conditions (p50, none, sr50 | G: p50, none, pre, pre_none) and differences p50 - none, p50 - sr50
(G: p50 - none, p50 - pre); inputs G, T, Z; vision modes V0, V1, V2; p_rule 0.8 / 0.5; all episodes and by A-2's
liked colour; plus D_swap(V1) - D_swap(V0).
Reading (fixed, Sec.44.3): applied to the main combination (p50, G, V1, p_rule 0.8, T_S 50). "CI excludes 0" = both
CIs exclude 0; "approximately 0" = both CIs include 0; if the two CIs disagree, no row is taken for that item.
-> Round20/analysis/{summary.json, tables.md, fig_round20_surprise_steps.{pdf,png}, fig_round20_dswap.{pdf,png}}
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import vdir_stageA as V  # noqa

EV = os.path.join(V.RES, 'paper_extra', 'Round20', 'eval')
OUT = os.path.join(V.RES, 'paper_extra', 'Round20', 'analysis')
DATA = os.path.join(V._PI, 'data', 'data')
PS = (80, 50)
VMS = ('V0', 'V1', 'V2')
VARS = ('N', 'O', 'S', 'R')
W = slice(53, 80)
PRE = slice(40, 50)
EPS = (160, 170, 180, 190, 200)
B = 1000
TZ = {'conds': ('p50', 'none', 'sr50'), 'pairs': tuple(range(6)), 'diffs': (('p50', 'none'), ('p50', 'sr50'))}
GG = {'conds': ('p50', 'none', 'pre', 'pre_none'), 'pairs': tuple(range(4)), 'diffs': (('p50', 'none'), ('p50', 'pre'))}
SUBSETS = ('all', 'red', 'green')


def files(fam, cond, s):
    eps = (0,) if cond in ('pre', 'pre_none') else EPS
    return [os.path.join(EV, f'{fam}_{cond}_s{s}_ep{e}.npz') for e in eps]


def episode_values(fam, cond, s, inp):
    """-> dict quantity -> (N,) episode values (mean over checkpoints) for every p / vision mode."""
    acc = {}
    fl = files(fam, cond, s)
    for f in fl:
        z = np.load(f)
        for p in PS:
            for vm in VMS:
                k = f'p{p}_{vm}_{inp}'
                if f'FP_{k}_N' in z:
                    sp = {v: z[f'FP_{k}_{v}'][:, W].astype(np.float64).mean(1) for v in VARS}
                    for nm, x in (('D_swap', sp['O'] - sp['N']), ('D_noswap', sp['R'] - sp['S']),
                                  *[(f'surprise_{v}', sp[v]) for v in VARS]):
                        acc.setdefault(f'{k}|{nm}', []).append(x)
                    acc.setdefault(f'{k}|surprise_steps', []).append(
                        np.stack([z[f'FP_{k}_{v}'].astype(np.float64).mean(0) for v in VARS]))    # (4, 80)
                if f'L1_{k}_N' in z and vm != 'V0':
                    l1 = {v: z[f'L1_{k}_{v}'][:, W].astype(np.float64).mean(1) for v in VARS}
                    acc.setdefault(f'{k}|self_err_dist', []).append((l1['N'] + l1['O']) / 2 - (l1['S'] + l1['R']) / 2)
                    acc.setdefault(f'{k}|self_err_noswap', []).append((l1['S'] + l1['R']) / 2)
                if f'TRK_{k}_N' in z:
                    tr = {v: z[f'TRK_{k}_{v}'].astype(np.float64) for v in VARS}
                    acc.setdefault(f'{k}|trk_dist', []).append((tr['N'][:, W].mean(1) + tr['O'][:, W].mean(1)) / 2 -
                                                               (tr['S'][:, W].mean(1) + tr['R'][:, W].mean(1)) / 2)
                    acc.setdefault(f'{k}|trk_pre', []).append(tr['S'][:, PRE].mean(1))
                    acc.setdefault(f'{k}|trk_post_noswap', []).append((tr['S'][:, W].mean(1) + tr['R'][:, W].mean(1)) / 2)
    return {k: np.mean(v, 0) for k, v in acc.items()}


def boot(X):
    """X (npairs, N) episode values -> 2.5 / 97.5 percentiles of the two-stage bootstrap mean."""
    rng = np.random.default_rng(0)
    npairs, N = X.shape
    pi = rng.integers(0, npairs, (B, npairs))
    ei = rng.integers(0, N, (B, npairs, N))
    m = X[pi[:, :, None], ei].mean(2).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def stat(X):
    """X (npairs, N) -> {'mean', 't': tsum of pair means, 'boot': CI, 'per_pair'}"""
    pm = X.mean(1)
    return {'mean': float(pm.mean()), 't': V.tsum(pm.tolist()), 'boot': boot(X), 'per_pair': pm.tolist()}


def collect():
    liked = h5py.File(os.path.join(DATA, 'paper_r20_swap_p80', 'data_N.h5'), 'r')['test']['a2_liked'][()]
    for p in PS:
        assert np.array_equal(h5py.File(os.path.join(DATA, f'paper_r20_swap_p{p}', 'data_N.h5'), 'r')['test']['a2_liked'][()], liked)
    sub = {'all': np.ones_like(liked, bool), 'red': liked == 1, 'green': liked == 0}
    EVV = {}           # (inp, cond) -> list over pairs of episode-value dicts (FP / tracking source + L1 source merged)
    for inp, spec in (('T', TZ), ('Z', TZ), ('G', GG)):
        for c in spec['conds']:
            lst = []
            for s in spec['pairs']:
                if inp == 'G':
                    d = episode_values('G', c, s, 'G')
                else:
                    d = episode_values('mse', c, s, inp)
                    d.update({k: v for k, v in episode_values('l1', c, s, inp).items() if 'self_err' in k})
                lst.append(d)
            EVV[(inp, c)] = lst
    return EVV, sub


def summarise(EVV, sub):
    S = {}
    curves = {}
    for inp, spec in (('G', GG), ('T', TZ), ('Z', TZ)):
        conds = spec['conds']
        for k in EVV[(inp, conds[0])][0]:
            if k.endswith('surprise_steps'):
                for c in conds:
                    curves[f'{inp}|{c}|{k}'] = np.stack([d[k] for d in EVV[(inp, c)]])     # (npairs, 4, 80)
                continue
            for c in conds:
                X = np.stack([d[k] for d in EVV[(inp, c)]])
                for sn, m in sub.items():
                    S[f'{inp}|{c}|{k}|{sn}'] = stat(X[:, m])
            for a, b in spec['diffs']:
                X = np.stack([d[k] for d in EVV[(inp, a)]]) - np.stack([d[k] for d in EVV[(inp, b)]])
                for sn, m in sub.items():
                    S[f'{inp}|{a}-{b}|{k}|{sn}'] = stat(X[:, m])
        # V1 - V0 of D_swap (same model, same episodes)
        for c in conds:
            for p in PS:
                X = np.stack([d[f'p{p}_V1_{inp}|D_swap'] - d[f'p{p}_V0_{inp}|D_swap'] for d in EVV[(inp, c)]])
                for sn, m in sub.items():
                    S[f'{inp}|{c}|p{p}_V1minusV0_{inp}|D_swap|{sn}'] = stat(X[:, m])
    return S, curves


def excl(st):
    """+1 / -1 if both CIs exclude 0 on that side, 0 if both include 0, None if they disagree."""
    t, b = st['t']['ci95'], st['boot']
    ex_t = 1 if t[0] > 0 else (-1 if t[1] < 0 else 0)
    ex_b = 1 if b[0] > 0 else (-1 if b[1] < 0 else 0)
    return ex_t if ex_t == ex_b else None


def reading(S):
    k = lambda inp, c, vm, q, p=80: S[f'{inp}|{c}|p{p}_{vm}_{inp}|{q}|all']   # noqa
    main = k('G', 'p50', 'V1', 'D_swap')
    ds, dn = excl(main), excl(k('G', 'p50', 'V1', 'D_noswap'))
    d_none = excl(S['G|p50-none|p80_V1_G|D_swap|all'])
    tT = excl(k('T', 'p50', 'V1', 'D_swap'))
    v10 = excl(S['G|p50|p80_V1minusV0_G|D_swap|all'])
    trk_d = k('G', 'p50', 'V1', 'trk_dist')['mean']
    trk_pre = k('G', 'p50', 'V1', 'trk_pre')['mean']
    rows = {
        'row1_location_bound': ds == -1,
        'row2_undecided_after_swap': ds == 0 and dn == 1,
        'row3_object_bound_experience': ds == 1 and d_none == 1,
        'row4_T_pos_G_le0': tT == 1 and main['mean'] <= 0,
        'row5_V0_eq_V1': v10 == 0,
    }
    return {'rows': rows, 'items': {'D_swap_main': ds, 'D_noswap_main': dn, 'D_swap_p50_minus_none_G': d_none,
                                    'D_swap_T_p50': tT, 'D_swap_V1_minus_V0_G': v10,
                                    'note_row3_sr50': 'G has no sign-randomised model; row 3 uses none only (T/Z rows give p50 - sr50)'},
            'tracking': {'disturbance': trk_d, 'pre_swap_error': trk_pre, 'disturbed': bool(trk_d >= 2 * trk_pre)}}


def fmt(st, d=4):
    t, b = st['t']['ci95'], st['boot']
    return f"{st['mean']:+.{d}f} [{t[0]:+.{d}f}, {t[1]:+.{d}f}] [{b[0]:+.{d}f}, {b[1]:+.{d}f}]"


def tables(S, R):
    L = ['# 第20弾 §2（探索・事前登録外）', '',
         '値：組の値の平均 [組の t の 95% CI] [2段のブートストラップの 95% CI]。G は 4 組、T・Z は 6 組。評価区間はステップ 53〜79。', '']
    for p in PS:
        for q, title in (('D_swap', '$D_{swap}$ = 驚き(O) − 驚き(N)'), ('D_noswap', '$D_{noswap}$ = 驚き(R) − 驚き(S)')):
            L += [f'## {title}、p_rule 0.{p // 10}', '', '| 入力 | 条件 | ' + ' | '.join(VMS) + ' |', '|---|---|---|---|---|']
            for inp, spec in (('G', GG), ('T', TZ), ('Z', TZ)):
                for c in list(spec['conds']) + [f'{a}-{b}' for a, b in spec['diffs']]:
                    L.append(f'| {inp} | {c} | ' + ' | '.join(fmt(S[f'{inp}|{c}|p{p}_{vm}_{inp}|{q}|all']) for vm in VMS) + ' |')
            L.append('')
        L += [f'## $D_{{swap}}$ の A-2 の好き別、p_rule 0.{p // 10}', '', '| 入力 | 条件 | 視覚 | 赤好き | 緑好き |', '|---|---|---|---|---|']
        for inp, spec in (('G', GG), ('T', TZ), ('Z', TZ)):
            for c in list(spec['conds']) + [f'{a}-{b}' for a, b in spec['diffs']]:
                for vm in VMS:
                    L.append(f'| {inp} | {c} | {vm} | ' + ' | '.join(fmt(S[f'{inp}|{c}|p{p}_{vm}_{inp}|D_swap|{sn}'])
                                                                    for sn in ('red', 'green')) + ' |')
        L += ['', f'## $D_{{swap}}$(V1) − $D_{{swap}}$(V0)、p_rule 0.{p // 10}', '', '| 入力 | 条件 | 値 |', '|---|---|---|']
        for inp, spec in (('G', GG), ('T', TZ), ('Z', TZ)):
            for c in spec['conds']:
                L.append(f'| {inp} | {c} | ' + fmt(S[f'{inp}|{c}|p{p}_V1minusV0_{inp}|D_swap|all']) + ' |')
        L += ['', f'## 驚き（評価区間の平均）、p_rule 0.{p // 10}', '', '| 入力 | 条件 | 視覚 | N | O | S | R |', '|---|---|---|---|---|---|---|']
        for inp, spec in (('G', GG), ('T', TZ), ('Z', TZ)):
            for c in spec['conds']:
                for vm in VMS:
                    L.append(f'| {inp} | {c} | {vm} | ' + ' | '.join(f"{S[f'{inp}|{c}|p{p}_{vm}_{inp}|surprise_{v}|all']['mean']:.4f}"
                                                                    for v in VARS) + ' |')
        L += ['', f'## 自己の誤差と追跡の乱れ（V1・V2）、p_rule 0.{p // 10}', '',
              '| 入力 | 条件 | 視覚 | 自己の誤差 入れ替えなし | 自己の誤差の乱れ（入れ替え − なし） | 追跡の誤差 入れ替え前 | 追跡 入れ替えなし（後半） | 追跡の乱れ（入れ替え − なし） |',
              '|---|---|---|---|---|---|---|---|']
        for inp, spec in (('G', GG), ('T', TZ), ('Z', TZ)):
            for c in list(spec['conds']) + [f'{a}-{b}' for a, b in spec['diffs']]:
                for vm in ('V1', 'V2'):
                    g = lambda q: S.get(f'{inp}|{c}|p{p}_{vm}_{inp}|{q}|all')   # noqa
                    L.append(f'| {inp} | {c} | {vm} | ' + ' | '.join(
                        (f"{g(q)['mean']:.4f}" if 'dist' not in q else fmt(g(q))) if g(q) else '—'
                        for q in ('self_err_noswap', 'self_err_dist', 'trk_pre', 'trk_post_noswap', 'trk_dist')) + ' |')
        L.append('')
    L += ['## 読み方の表（主の組み合わせ：経験50%・入力 G・V1・p_rule 0.8・$T_s$ = 50）', '',
          '```', json.dumps(R, ensure_ascii=False, indent=1), '```', '']
    txt = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'tables.md'), 'w').write(txt)
    print(txt)


def figures(S, curves):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt
    # 1. main combination: surprise per step, N and O (S, R dashed), mean over pairs
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, p in zip(axs, PS):
        C = curves[f'G|p50|p{p}_V1_G|surprise_steps']                 # (4 pairs, 4 vars, 80)
        st = np.arange(80)
        for j, (v, col, ls) in enumerate(zip(VARS, ('C0', 'C1', 'C0', 'C1'), ('-', '-', ':', ':'))):
            ax.plot(st[30:], C[:, j, 30:].mean(0), color=col, ls=ls, lw=1.6,
                    label={'N': 'N (new location)', 'O': 'O (old location)', 'S': 'S (no swap, liked)', 'R': 'R (no swap, disliked)'}[v])
        ax.axvspan(49.5, 52.5, color='0.85', zorder=0)
        ax.axvspan(52.5, 79.5, color='0.95', zorder=0)
        ax.set_xlabel('step t (input frame t, target t+1)'); ax.set_ylabel('FP (other-side feature error)')
        ax.set_title(f'50% experience, input G, V1, p_rule 0.{p // 10} (4 pairs)', fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round20_surprise_steps.{ext}'), dpi=150)
    plt.close(fig)
    # 2. D_swap per condition: pair points and mean with t CI
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    for ax, p in zip(axs, PS):
        labs, ys, y = [], [], 0
        for inp, spec in (('G', GG), ('T', TZ), ('Z', TZ)):
            for c in spec['conds']:
                for vm in VMS:
                    st = S[f'{inp}|{c}|p{p}_{vm}_{inp}|D_swap|all']
                    ax.plot(st['per_pair'], [y] * len(st['per_pair']), 'o', ms=3, color='0.5', alpha=0.7)
                    t = st['t']['ci95']
                    ax.errorbar(st['mean'], y, xerr=[[st['mean'] - t[0]], [t[1] - st['mean']]], fmt='D', ms=4,
                                color='C3' if (inp, c, vm) == ('G', 'p50', 'V1') else 'k', capsize=2)
                    labs.append(f'{inp} {c} {vm}')
                    ys.append(y)
                    y -= 1
            y -= 0.6
        ax.axvline(0, color='0.3', lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(labs, fontsize=6)
        ax.set_xlabel('D_swap = surprise(O) - surprise(N)')
        ax.set_title(f'p_rule 0.{p // 10}: pairs (grey) and mean with pair t 95% CI', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_round20_dswap.{ext}'), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    EVV, sub = collect()
    S, curves = summarise(EVV, sub)
    R = reading(S)
    json.dump({'stats': S, 'reading': R, 'curves_main': {k: v.mean(0).tolist() for k, v in curves.items() if k.startswith('G|p50|')}},
              open(os.path.join(OUT, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
    tables(S, R)
    figures(S, curves)


if __name__ == '__main__':
    main()
