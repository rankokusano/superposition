"""
Paper round 11 E1 (docs/paper_extra_log.md Sec.35.5, pre-registered before training; addendum Sec.35.9 before the
evaluation): 50% sign-randomised (signrand50, paper_extra/E1_signrand50) vs the same pair's 50% run (v12_s1_mixed),
pairs 1-6. CPU, saved per-episode values only.
  pair value : mean over the 5 late points of each model's nanmean (main form); episode-paired form = mean over the
               5 points of the mean difference over episodes read in both (eval_g5_v12.summ). Across pairs
               across_pairs_v13.test (t CI df 5, two-sided sign test).
  main       : d_P = (g_hold - g_flip)/2 (G3, K = 15, mse late 5), signrand50 - 50%, 6-pair CI upper < 0.
  secondary  : sign-following parts (1) green, (2) pref (l1 late 5), signrand50 - 50%, CI upper < 0;
               E2 U_pref (pref, FP, frames 1-50, mse late 5), 50% - signrand50, CI lower > 0 (e2_aggregate.comparison);
               U^pos_pref (Sec.35.9) is computed by e2pos_aggregate.py (summary.json 'E1') and copied here if present.
  per pair   : points whose episode-paired bootstrap CI (1000, seed 0) excludes 0 in the expected direction, 3 of 5.
  description: G5b; signrand50 - none for G5b / d_P / sign-following; g hold / zero / flip; readable fractions;
               tracking (G1 visible one-step-ahead R^2, mse late 5) >= 0.80 per pair; 80% sign-randomised (round 6)
               alongside; E2 U_pref / U_use of signrand50 alone and 50% - signrand50 in all 16 cells.
  two forms  : main form and episode-paired form; disagreement (CI excludes 0 in one only, or per-pair signs differ)
               is flagged (Sec.34.3.5).
  exclusion  : outside the pre-registration (supplement S5 style): orig / a (episodes read in none, 50%, signrand50)
               / b1 (unread -> 0 in both) / b2 (unread -> 0 in 50% only), G5b and sign-following (1)(2).
  checks     : signrand50 rep bit-identity (trial, G5, E2 both stages); 50% g3flip hold / zero = saved G3 (s1eval
               per_ep); 50% d_P of pairs 1-3 = B7 (Sec.31); d_P NaN pattern identical across signrand50 / 50% / none;
               checkpoints saved.
-> data/result/paper_extra/E1_signrand50/summary.json, tables.md
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_PI, 'analyze', 'v13')):
    sys.path.insert(0, p)
import c_aggregate as CA  # noqa
import r4_exclusion as R4  # noqa
import e2_aggregate as E2A  # noqa
from across_pairs_v13 import test as ttest  # noqa

RES = CA.RES
EPS = CA.EPS
summ = CA.summ
E1 = os.path.join(RES, 'paper_extra', 'E1_signrand50')
EV = os.path.join(E1, 'eval')
SEEDS = range(6)
MODELS = ('sr50', '50', 'none')
NAME = {'sr50': '符号無作為（経験50%）', '50': '経験50%', 'none': '経験なし'}
R4.M['sr50'] = ('paper_extra/E1_signrand50/eval/g5/signrand50', 'signrand50',
                'paper_extra/E1_signrand50/eval/decomp/v13/post/g5a_decomp', 'signrand50')
QN = {'dP': 'd_P', 'signfollow_green': '符号に従う分 ①（緑好み）', 'signfollow_pref': '符号に従う分 ②（好み混合）',
      'G5b': 'G5b', 'G3hold': 'g 保つ'}


def g3(m, s, e, q):
    if m == 'none':
        return np.load(os.path.join(RES, 'paper_extra', 'C_signrand', 'eval', 'g3flip', f'random_s{s}_ep{e}.npz'))[q]
    return np.load(os.path.join(EV, 'g3flip', f'{"signrand50" if m == "sr50" else "mixed50"}_s{s}_ep{e}.npz'))[q]


def val(m, s, e, q):
    if q == 'dP':
        return g3(m, s, e, 'dP')
    if q == 'G3hold':
        return g3(m, s, e, 'hold')
    return np.asarray(R4.val(m, s, e, {'G5b': 'G5b', 'signfollow_green': 'green', 'signfollow_pref': 'pref'}[q]), float)


def trial_json(m, s, e, tag=''):
    if m == 'sr50':
        f = os.path.join(EV, 's1eval', 'signrand50', f's{s}', f'eval_signrand50_ep{e}{tag}.json')
    elif m == '50':
        f = os.path.join(RES, 'v12', 'main', 's1eval', f's{s}', f'eval_mixed_ep{e}.json') if s <= 3 else \
            os.path.join(RES, 'v13', 'v12ref', 's1eval', f's{s}', f'eval_mixed_ep{e}.json')
    else:
        f = os.path.join(RES, 'v13', 'main', 's1eval', f's{s}', f'eval_random_ep{e}.json')
    return json.load(open(f))


def model_mean(m, s, q):
    return float(np.mean([np.nanmean(val(m, s, e, q)) for e in EPS]))


def paired_mean(a, b, s, q):
    return float(np.mean([summ(val(a, s, e, q) - val(b, s, e, q))['mean'] for e in EPS]))


def per_pair(a, b, s, q, sign):
    pts = [summ(val(a, s, e, q) - val(b, s, e, q)) for e in EPS]
    ok = [CA.neg(p) if sign < 0 else CA.pos(p) for p in pts]
    return {'points': pts, 'n_ok': int(sum(ok)), 'pair_ok': sum(ok) >= 3}


def across(a, b, q, sign):
    """6-pair t of a - b in both forms; holds = CI excludes 0 in the expected direction (sign -1: upper < 0)."""
    dm = [model_mean(a, s, q) - model_mean(b, s, q) for s in SEEDS]
    dp = [paired_mean(a, b, s, q) for s in SEEDS]
    tm, tp = ttest(dm), ttest(dp)
    hold = (lambda t: t['ci95'][1] < 0) if sign < 0 else (lambda t: t['ci95'][0] > 0)
    pp = [per_pair(a, b, s, q, sign) for s in SEEDS]
    excl = lambda t: t['ci95'][0] > 0 or t['ci95'][1] < 0  # noqa
    return {'diff_means': dm, 'diff_paired': dp, 't_means': tm, 't_paired': tp,
            'holds_main_form': hold(tm), 'holds_paired_form': hold(tp),
            'n_expected_dir_means': int(sum((x < 0) if sign < 0 else (x > 0) for x in dm)),
            'n_expected_dir_paired': int(sum((x < 0) if sign < 0 else (x > 0) for x in dp)),
            'forms_disagree': excl(tm) != excl(tp) or [x < 0 for x in dm] != [x < 0 for x in dp],
            'per_pair': pp, 'pairs_ok': int(sum(p['pair_ok'] for p in pp)), 'points_ok': [p['n_ok'] for p in pp]}


def values(s):
    lv = {}
    for m in MODELS:
        d = {q: [float(np.nanmean(val(m, s, e, q))) for e in EPS] for q in ('dP', 'G5b', 'signfollow_green', 'signfollow_pref')}
        for q in ('hold', 'zero', 'flip'):
            d[f'G3_{q}'] = [float(np.nanmean(g3(m, s, e, q))) for e in EPS]
        js = [trial_json(m, s, e) for e in EPS]
        d['G1_visible'] = [j['tracking']['p_next']['visible'] for j in js]
        d['G1_hidden'] = [j['tracking']['p_next']['hidden'] for j in js]
        d['self_err_visible'] = [j['err_mean_visible'] for j in js]
        d['self_err_hidden'] = [j['err_mean_hidden'] for j in js]
        d['read_dP'] = [float(np.isfinite(g3(m, s, e, 'dP')).mean()) for e in EPS]
        for q in ('G5b', 'signfollow_green', 'signfollow_pref'):
            d[f'read_{q}'] = [float(np.isfinite(val(m, s, e, q)).mean()) for e in EPS]
        d['tracking_late5'] = float(np.mean(d['G1_visible']))
        lv[m] = d
    return lv


def checks(s):
    ck = {}
    a = np.load(os.path.join(EV, 'g5', 'signrand50', f's{s}', 'g5_per_ep200.npz'))
    b = np.load(os.path.join(EV, 'g5_rep', 'signrand50', f's{s}', 'g5_per_ep200.npz'))
    ck['g5_rep'] = all(np.array_equal(a[q], b[q], equal_nan=True) for q in a.files)
    ck['trial_rep'] = trial_json('sr50', s, 200) == trial_json('sr50', s, 200, '_rep')
    e2 = []
    for k in E2A.SETS:
        for st in ('mse', 'l1'):
            x = np.load(os.path.join(E2A.RUNS, f'sr50_{st}_s{s}_ep200_{k}.npz'))
            y = np.load(os.path.join(E2A.RUNS, f'sr50_{st}_s{s}_ep200_{k}_rep.npz'))
            e2.append(all(np.array_equal(x[q], y[q]) for q in x.files))
    ck['e2_rep'] = all(e2)
    sv = os.path.join(RES, 'v12', 'main', 's1eval', f's{s}') if s <= 3 else os.path.join(RES, 'v13', 'v12ref', 's1eval', f's{s}')
    ck['g3flip_50_equals_saved'] = all(
        np.array_equal(g3('50', s, e, q), np.load(os.path.join(sv, f'per_ep{e}.npz'))[f'mixed__freeze_K15_{q}'], equal_nan=True)
        for e in EPS for q in ('hold', 'zero'))
    if s <= 2:
        ck['dP_50_equals_B7'] = all(
            np.array_equal(g3('50', s, e, q), np.load(os.path.join(RES, 'paper_extra', 'B7', f'mixed50_s{s}_ep{e}.npz'))[q], equal_nan=True)
            for e in EPS for q in ('hold', 'dP'))
    ck['dP_nan_pattern_same'] = all(
        np.array_equal(np.isnan(g3('sr50', s, e, q)), np.isnan(g3(m, s, e, q)))
        for m in ('50', 'none') for e in EPS for q in ('hold', 'flip', 'dP'))
    ck['ckpts'] = {'mse': len(os.listdir(os.path.join(E1, 's1_signrand50_mse', str(s), 'model'))),
                   'l1': len(os.listdir(os.path.join(E1, f's1_signrand50_l1_s{s}', str(s), 'model')))}
    return ck


def variant(s, e, q, v):
    X = {m: np.array(val(m, s, e, q), float) for m in ('sr50', '50')}
    if v == 'a':
        ok = np.all([np.isfinite(val(m, s, e, q)) for m in MODELS], 0)
        for m in X:
            X[m][~ok] = np.nan
    elif v == 'b1':
        for m in X:
            X[m][np.isnan(X[m])] = 0.0
    elif v == 'b2':
        X['50'][np.isnan(X['50'])] = 0.0
    return X


def exclusion():
    out = {}
    for q in ('G5b', 'signfollow_green', 'signfollow_pref'):
        for v in ('orig', 'a', 'b1', 'b2'):
            pp, dm, dp = {}, [], []
            for s in SEEDS:
                Xs = [variant(s, e, q, v) for e in EPS]
                pts = [summ(X['sr50'] - X['50']) for X in Xs]
                ok = [CA.neg(p) for p in pts]
                pp[f's{s}'] = {'n_ok': int(sum(ok)), 'pair_ok': sum(ok) >= 3}
                dm.append(float(np.mean([np.nanmean(X['sr50']) - np.nanmean(X['50']) for X in Xs])))
                dp.append(float(np.mean([p['mean'] for p in pts])))
            out[f'{q}|{v}'] = {'per_pair': pp, 'pairs_ok': sum(p['pair_ok'] for p in pp.values()),
                               't_means': ttest(dm), 't_paired': ttest(dp), 'diff_means': dm, 'diff_paired': dp}
    return out


def e2_block():
    B = {}
    for k in E2A.SETS:
        for q in E2A.ERRS:
            for w in E2A.WINS:
                cell = f'{k}|{q}|{w}'
                B[cell] = {'p50_minus_sr50': E2A.comparison('p50', 'sr50', k, q, w),
                           'sr50_alone': {kind: [E2A.pair_value('sr50', s, k, q, w, kind) for s in SEEDS] for kind in ('pref', 'use')}}
                for kind in ('pref', 'use'):
                    B[cell]['sr50_alone'][f'{kind}_t'] = ttest(B[cell]['sr50_alone'][kind])
    return B


def f3(r):
    return f"{r['mean']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"


def tables(R):
    L = ['# E1 符号無作為（経験50%）の集計（`analyze/paper_extra/e1_aggregate.py`、事前登録 §35.5・§35.9）', '']
    L += ['## 表1 6 組の判定（差 = 符号無作為（経験50%）− 経験50%。E2 と位置のずれは 経験50% − 符号無作為（経験50%））', '',
          '| 量 | 判定の向き | 主の形 平均 [95% CI] | 期待の向きの組 | 符号検定 p | 対の形 平均 [95% CI] | 組ごと（3/5 点） | 各組の点 | 形の食い違い | 判定 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    J = R['judgement']
    for key, lab, dirn in (('dP', '主：d_P', '上限 < 0'), ('signfollow_green', '副：符号に従う分 ①（緑好み）', '上限 < 0'),
                           ('signfollow_pref', '副：符号に従う分 ②（好み混合）', '上限 < 0'), ('G5b', '記述：G5b', '（判定しない）')):
        a = R['across'][key]
        L.append(f"| {lab} | {dirn} | {f3(a['t_means'])} | {a['n_expected_dir_means']}/6 | {a['t_means']['sign_p']:.3f} | "
                 f"{f3(a['t_paired'])} | {a['pairs_ok']}/6 | {a['points_ok']} | {'あり' if a['forms_disagree'] else 'なし'} | "
                 f"{('成り立つ' if a['holds_main_form'] else '成り立たない') if key != 'G5b' else '—'} |")
    e = R['e2']['pref|FP|1-50']['p50_minus_sr50']
    L.append(f"| 副：E2 の U_pref（好み混合・FP・1〜50） | 下限 > 0 | {f3(e['t'])} | {e['n_pos']}/6 | {e['t']['sign_p']:.3f} | "
             f"（主の形と同じ） | {e['pairs_ok']}/6 | {[p['n_ok'] for p in e['per_pair']]} | — | {'成り立つ' if e['holds'] else '成り立たない'} |")
    if R.get('pos') is not None:
        p, pj = R['pos']['pref'], R['pos']['judgement']
        excl = lambda t: t['ci95'][0] > 0 or t['ci95'][1] < 0  # noqa
        dis = excl(p['t']) != excl(p['t_paired']) or [x > 0 for x in p['diffs']] != [x > 0 for x in p['diffs_paired']]
        L.append(f"| 副：位置のずれの U^pos_pref（好み混合・10〜50） | 下限 > 0 | {f3(p['t'])} | {p['n_pos']}/6 | "
                 f"{p['t']['sign_p']:.3f} | {f3(p['t_paired'])} | {p['pairs_ok']}/6 | "
                 f"{[x['n_ok'] for x in p['per_pair']]} | {'あり' if dis else 'なし'} | "
                 f"{'成り立つ' if pj['holds'] else '成り立たない'} |")
    else:
        L.append('| 副：位置のずれの U^pos_pref（好み混合・10〜50） | 下限 > 0 | 未計算 | | | | | | | 未計算 |')
    L += ['', f"- 主の判定（d_P）：{'成り立つ' if J['main'] else '成り立たない'}（主の形）。対の形：{'成り立つ' if J['main_paired_form'] else '成り立たない'}。", '']
    L += ['## 表2 組ごとの値（最終5点の平均）', '',
          '| 組 | 量 | 符号無作為（経験50%） | 経験50% | 経験なし | 差（sr50 − 50%） |', '|---|---|---|---|---|---|']
    for s in SEEDS:
        V = R['values'][f's{s}']
        for q in ('dP', 'signfollow_green', 'signfollow_pref', 'G5b', 'G3_hold', 'G3_flip', 'G1_visible'):
            m = {x: float(np.mean(V[x][q])) for x in MODELS}
            L.append(f"| {s + 1} | {QN.get(q, q)} | {m['sr50']:+.4f} | {m['50']:+.4f} | {m['none']:+.4f} | {m['sr50'] - m['50']:+.4f} |")
    L += ['', '## 表3 操作の確認：共通テストの追跡（G1 の見える区間の一歩先、MSE 期間の最終5点の平均）', '',
          '| 組 | 符号無作為（経験50%） | 経験50% | 経験なし | ≥ 0.80 |', '|---|---|---|---|---|']
    for s in SEEDS:
        V = R['values'][f's{s}']
        L.append(f"| {s + 1} | {V['sr50']['tracking_late5']:.4f} | {V['50']['tracking_late5']:.4f} | {V['none']['tracking_late5']:.4f} | "
                 f"{'満たす' if V['sr50']['tracking_late5'] >= 0.80 else '満たさない'} |")
    L += ['', '## 表4 記述：符号無作為（経験50%）− 経験なし（6 組、主の形）と、経験80% の符号無作為（第6弾、符号無作為 − 経験80%）', '',
          '| 量 | 経験50%：sr50 − 経験なし | 経験50%：sr50 − 経験50% | 経験80%：sr80 − 経験80%（第6弾） |', '|---|---|---|---|']
    for q in ('dP', 'signfollow_green', 'signfollow_pref', 'G5b', 'G3hold'):
        r6 = R['round6'].get(q)
        L.append(f"| {QN[q]} | {f3(R['across_vsnone'][q]['t_means'])} | {f3(R['across'][q]['t_means']) if q in R['across'] else '—'} | "
                 f"{f3(r6['all6_means']) if r6 else '—'} |")
    L += ['', '## 表5 読めた割合（最終5点の平均、6 組の範囲）', '', '| 量 | 符号無作為（経験50%） | 経験50% | 経験なし |', '|---|---|---|---|']
    for q in ('read_dP', 'read_G5b', 'read_signfollow_green', 'read_signfollow_pref'):
        rg = {m: [float(np.mean(R['values'][f's{s}'][m][q])) for s in SEEDS] for m in MODELS}
        L.append(f"| {q} | " + ' | '.join(f"{min(rg[m]):.3f}〜{max(rg[m]):.3f}" for m in MODELS) + ' |')
    L += ['', '## 表6 E2：経験50% − 符号無作為（経験50%）の U_pref と、符号無作為（経験50%）単独（全 16 通り。主は好み混合・FP・1〜50）', '',
          '| データ | 誤差 | 区間 | 50% − sr50 平均 [95% CI] | 正の組 | 組ごと | sr50 単独 U_pref | sr50 単独 U_use |', '|---|---|---|---|---|---|---|---|']
    for cell, B in R['e2'].items():
        k, q, w = cell.split('|')
        c = B['p50_minus_sr50']
        L.append(f"| {E2A.SETNAME[k]} | {q} | {w} | {f3(c['t'])} | {c['n_pos']}/6 | {c['pairs_ok']}/6 | "
                 f"{f3(B['sr50_alone']['pref_t'])} | {f3(B['sr50_alone']['use_t'])} |")
    L += ['', '## 表7 事前登録外：除外の作り（補足資料 S5 と同じ。sr50 − 50%、6 組の t）', '',
          '| 量 | 作り | 主の形 平均 [95% CI] | 対の形 平均 [95% CI] | 組ごと（3/5 点） |', '|---|---|---|---|---|']
    for key, x in R['exclusion'].items():
        q, v = key.split('|')
        L.append(f"| {QN[q]} | {v} | {f3(x['t_means'])} | {f3(x['t_paired'])} | {x['pairs_ok']}/6 |")
    L += ['', '## 確認', '', '```', json.dumps(R['checks'], ensure_ascii=False), f"all_checks_ok: {R['all_checks_ok']}", '```', '']
    return '\n'.join(L)


def main():
    R = {'values': {f's{s}': values(s) for s in SEEDS}, 'checks': {f's{s}': checks(s) for s in SEEDS}}
    R['across'] = {q: across('sr50', '50', q, -1) for q in ('dP', 'signfollow_green', 'signfollow_pref', 'G5b', 'G3hold')}
    R['across_vsnone'] = {q: across('sr50', 'none', q, 1) for q in ('dP', 'signfollow_green', 'signfollow_pref', 'G5b', 'G3hold')}
    R['across_50_vsnone'] = {q: across('50', 'none', q, 1) for q in ('dP', 'signfollow_green', 'signfollow_pref', 'G5b', 'G3hold')}
    R['e2'] = e2_block()
    pf = os.path.join(RES, 'paper_extra', 'E2', 'pos', 'summary.json')
    P = json.load(open(pf)).get('E1') if os.path.exists(pf) else None
    R['pos'] = P
    r6 = json.load(open(os.path.join(RES, 'paper_extra', 'C_signrand', 's3s5', 'summary.json')))['across6']
    R['round6'] = {q: r6[q] for q in ('dP', 'signfollow_green', 'signfollow_pref', 'G5b', 'G3hold') if q in r6}
    R['exclusion'] = exclusion()
    R['tracking_ok'] = {f's{s}': R['values'][f's{s}']['sr50']['tracking_late5'] >= 0.80 for s in SEEDS}
    R['judgement'] = {'main': R['across']['dP']['holds_main_form'], 'main_paired_form': R['across']['dP']['holds_paired_form'],
                      'signfollow_green': R['across']['signfollow_green']['holds_main_form'],
                      'signfollow_pref': R['across']['signfollow_pref']['holds_main_form'],
                      'e2_upref': R['e2']['pref|FP|1-50']['p50_minus_sr50']['holds'],
                      'pos_upref': (P['judgement']['holds'] if P is not None else '未計算')}
    flat = []
    for c in R['checks'].values():
        flat += [v for k, v in c.items() if k != 'ckpts']
        flat.append(c['ckpts'] == {'mse': 201, 'l1': 21})
    R['all_checks_ok'] = all(flat)
    with open(os.path.join(E1, 'summary.json'), 'w') as f:
        json.dump(R, f, indent=1, ensure_ascii=False)
    T = tables(R)
    with open(os.path.join(E1, 'tables.md'), 'w') as f:
        f.write(T)
    print(T)


if __name__ == '__main__':
    main()
