"""
Paper round 11 E2 aggregation (docs/paper_extra_log.md Sec.35.3, pre-registered in cdd5043). CPU, saved runs only
(e2_run.py -> data/result/paper_extra/E2/runs/).
Per episode: U_pref = L(opp) - L(true), U_use = L(zero) - L(true), L = mean of the error over the window
(frames 1-50 = t 0..49, or 1-100 = t 0..99); FP from the MSE-stage points, L1 from the L1-stage points.
Pair value = mean over the 5 late points of the episode mean (no unreadable episodes, so the model-mean difference and
the episode-paired difference are the same number). Across pairs: across_pairs_v13.test (t CI df 5, sign test).
Per pair (description): points whose episode-paired bootstrap CI (eval_g5_v12.summ, 1000, seed 0) is > 0, 3 of 5.
  main : pref / FP / 1-50, p50 - none, 6-pair CI lower > 0
  S1   : p50 alone > 0;  S2 : p80 - none > 0;  S3 : p80 - sr80 > 0   (all 16 set/error/window cells, main's too)
  desc : every model alone (U_pref, U_use), liked Red / Green split on the two preference-mixed sets
  checks: ep200 run twice bit-identical; overlap with the saved matching runs (v14/inverse, C_signrand/eval/inverse)
-> data/result/paper_extra/E2/summary.json, tables.md
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_PI, 'analyze', 'v13'), os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)
from across_pairs_v13 import test as ttest  # noqa
from eval_g5_v12 import summ  # noqa
from eval_trial_v12 import RES  # noqa

D = os.path.join(RES, 'paper_extra', 'E2')
RUNS = os.path.join(D, 'runs')
FAMS = ('none', 'p50', 'p80', 'sr80')
NAME = {'none': '経験なし', 'p50': '経験50%', 'p80': '経験80%', 'sr80': '符号無作為（経験80%）'}
SEEDS = range(6)
EPS = (160, 170, 180, 190, 200)
SETS = ('pref', 'green', 'pref_a2p80', 'green_a2p80')
SETNAME = {'pref': '好み混合', 'green': '緑好み', 'pref_a2p80': 'A-2 0.8 好み混合', 'green_a2p80': 'A-2 0.8 緑好み'}
ERRS = {'FP': 'mse', 'L1': 'l1'}
WINS = {'1-50': 50, '1-100': 100}
COMPS = {'main_p50_none': ('p50', 'none'), 'S1_p50': ('p50', None), 'S2_p80_none': ('p80', 'none'), 'S3_p80_sr80': ('p80', 'sr80')}
MAIN = ('pref', 'FP', '1-50')
_cache = {}


def load(f, s, e, k, q, tag=''):
    key = (f, s, e, k, q, tag)
    if key not in _cache:
        _cache[key] = np.load(os.path.join(RUNS, f'{f}_{ERRS[q]}_s{s}_ep{e}_{k}{tag}.npz'))
    return _cache[key]


def U(f, s, e, k, q, w, kind='pref'):
    z, n = load(f, s, e, k, q), WINS[w]
    base = z[f'{q}_true'][:, :n].mean(1)
    return z[f'{q}_{"opp" if kind == "pref" else "zero"}'][:, :n].mean(1) - base


def pair_value(f, s, k, q, w, kind='pref', mask=None):
    return float(np.mean([np.mean(U(f, s, e, k, q, w, kind)[mask] if mask is not None else U(f, s, e, k, q, w, kind)) for e in EPS]))


def comparison(a, b, k, q, w):
    diffs, pts = [], []
    for s in SEEDS:
        d = pair_value(a, s, k, q, w) - (pair_value(b, s, k, q, w) if b else 0.0)
        diffs.append(d)
        p = [summ(U(a, s, e, k, q, w) - (U(b, s, e, k, q, w) if b else 0.0)) for e in EPS]
        ok = [x['mean'] > 0 and x['ci95'][0] > 0 for x in p]
        pts.append({'n_ok': int(sum(ok)), 'pair_ok': sum(ok) >= 3, 'points': p})
    t = ttest(diffs)
    return {'diffs': diffs, 't': t, 'holds': t['ci95'][0] > 0, 'n_pos': int(sum(d > 0 for d in diffs)),
            'per_pair': pts, 'pairs_ok': int(sum(p['pair_ok'] for p in pts))}


def checks():
    out = {'rep_identical': {}, 'saved_overlap': {}}
    for f in FAMS:
        for s in SEEDS:
            for k in SETS:
                for q, st in ERRS.items():
                    a, b = load(f, s, 200, k, q), load(f, s, 200, k, q, '_rep')
                    out['rep_identical'][f'{f}_{st}_s{s}_{k}'] = all(np.array_equal(a[x], b[x]) for x in a.files)
    inv = {'none': ('v14', 'v12_random', 'v12mse_random'), 'p50': ('v14', 'v12_mixed', 'v12mse_mixed'),
           'p80': ('v14', 'v13_mixed', 'v13mse_mixed'), 'sr80': ('csr', 'csr_signrand', 'csrmse_signrand')}
    root = {'v14': os.path.join(RES, 'v14', 'inverse'), 'csr': os.path.join(RES, 'paper_extra', 'C_signrand', 'eval', 'inverse')}
    for f, (r, l1n, msen) in inv.items():
        for s in SEEDS:
            for k in ('pref', 'green'):
                for q, nm in (('FP', msen), ('L1', l1n)):
                    for e in EPS:
                        p = os.path.join(root[r], f'{nm}_{s}_ep{e}_{k}.npz')
                        if not os.path.exists(p):
                            continue
                        z, n = np.load(p), load(f, s, e, k, q)
                        red = (z['a2_liked'] == 1)[:, None]
                        ref = {'true': np.where(red, z[f'{q}_ref_red'], z[f'{q}_ref_green']),
                               'opp': np.where(red, z[f'{q}_ref_green'], z[f'{q}_ref_red']), 'zero': z[f'{q}_c0']}
                        out['saved_overlap'][f'{f}_{q}_s{s}_ep{e}_{k}'] = {
                            'identical': all(np.array_equal(n[f'{q}_{c}'], ref[c]) for c in ref),
                            'max_abs': float(max(np.abs(n[f'{q}_{c}'] - ref[c]).max() for c in ref))}
    ov = out['saved_overlap'].values()
    out['rep_all'] = all(out['rep_identical'].values())
    out['saved_n'] = len(out['saved_overlap'])
    out['saved_identical_n'] = sum(v['identical'] for v in ov)
    out['saved_max_abs'] = max(v['max_abs'] for v in ov) if ov else None
    return out


def main():
    R = {'cells': {}, 'alone': {}, 'liked_split': {}}
    for k in SETS:
        for q in ERRS:
            for w in WINS:
                cell = f'{k}|{q}|{w}'
                R['cells'][cell] = {c: comparison(a, b, k, q, w) for c, (a, b) in COMPS.items()}
                R['alone'][cell] = {f: {kind: [pair_value(f, s, k, q, w, kind) for s in SEEDS] for kind in ('pref', 'use')}
                                    for f in FAMS}
                for f in FAMS:
                    R['alone'][cell][f]['pref_t'] = ttest(R['alone'][cell][f]['pref'])
                    R['alone'][cell][f]['use_t'] = ttest(R['alone'][cell][f]['use'])
                if k.startswith('pref'):
                    lk = load('none', 0, 160, k, q)['a2_liked']
                    R['liked_split'][cell] = {f: {nm: [pair_value(f, s, k, q, w, mask=(lk == v)) for s in SEEDS]
                                                  for nm, v in (('red', 1), ('green', 0))} for f in FAMS}
    m = R['cells']['|'.join(MAIN)]['main_p50_none']
    R['main'] = {'cell': MAIN, 'diff_mean': m['t']['mean'], 'ci95': m['t']['ci95'], 'holds': m['holds'],
                 'n_pos': m['n_pos'], 'sign_p': m['t']['sign_p'], 'pairs_ok': m['pairs_ok']}
    R['checks'] = checks()
    with open(os.path.join(D, 'summary.json'), 'w') as f:
        json.dump(R, f, indent=1)
    write_tables(R)


def ci(t):
    return f'{t["mean"]:+.4f} [{t["ci95"][0]:+.4f}, {t["ci95"][1]:+.4f}]'


def write_tables(R):
    L = ['## 主の判定（好み混合・FP・フレーム1〜50、経験50% − 経験なしの U_pref）', '']
    m = R['cells']['|'.join(MAIN)]['main_p50_none']
    L += [f'- 6組の差：{ci(m["t"])}、正の組 {m["n_pos"]}/6、符号検定 p = {m["t"]["sign_p"]:.3f}、組ごとの記述で正の組 {m["pairs_ok"]}/6 → '
          f'**{"成立（下限 > 0）" if m["holds"] else ("逆向き（上限 < 0）" if m["t"]["ci95"][1] < 0 else "不成立（CI が 0 を含む）")}**', '']
    cell = '|'.join(MAIN)
    L += ['## 表1 組ごとの U_pref と U_use（好み混合・FP・フレーム1〜50、最終5点の平均）', '',
          '| 条件 | 量 | ' + ' | '.join(f'組{s + 1}' for s in SEEDS) + ' | 6組の平均 [95% CI] |', '|' + '---|' * 9]
    for f in FAMS:
        for kind, nm in (('pref', 'U_pref'), ('use', 'U_use')):
            v = R['alone'][cell][f][kind]
            L.append(f'| {NAME[f]} | {nm} | ' + ' | '.join(f'{x:+.4f}' for x in v) + f' | {ci(R["alone"][cell][f][kind + "_t"])} |')
    for c, (a, b) in COMPS.items():
        if b is None:
            continue
        r = R['cells'][cell][c]
        L.append(f'| {NAME[a]} − {NAME[b]} | U_pref の差 | ' + ' | '.join(f'{x:+.4f}' for x in r['diffs']) + f' | {ci(r["t"])} |')
        L.append(f'| {NAME[a]} − {NAME[b]} | 組ごとの点（5点中） | ' + ' | '.join(str(p['n_ok']) for p in r['per_pair']) + ' | |')
    r = R['cells'][cell]['S1_p50']
    L.append('| 経験50% 単独 | 組ごとの点（5点中） | ' + ' | '.join(str(p['n_ok']) for p in r['per_pair']) + ' | |')
    L += ['', '## 表2 全16通り（6組の t の 95% CI、自由度5。判定：主と S1〜S3 は下限 > 0）', '',
          '| データ | 誤差 | 区間 | 経験50% − 経験なし | S1 経験50% 単独 | S2 経験80% − 経験なし | S3 経験80% − 符号無作為 |', '|---|---|---|---|---|---|---|']
    for k in SETS:
        for q in ERRS:
            for w in WINS:
                c = R['cells'][f'{k}|{q}|{w}']
                cells = [f'{ci(c[x]["t"])} {"○" if c[x]["holds"] else "×"}（正 {c[x]["n_pos"]}/6、組 {c[x]["pairs_ok"]}/6）' for x in COMPS]
                L.append(f'| {SETNAME[k]} | {q} | {w} | ' + ' | '.join(cells) + ' |')
    L += ['', '○ = 下限 > 0、× = 下限 ≤ 0。「正」は差が正の組の数、「組」は組ごとの記述（5点中3点）で正の組の数。', '',
          '## 表3 各条件単独の U_pref・U_use（6組の平均 [95% CI]、全16通り）', '',
          '| データ | 誤差 | 区間 | ' + ' | '.join(f'{NAME[f]} U_pref' for f in FAMS) + ' | ' + ' | '.join(f'{NAME[f]} U_use' for f in FAMS) + ' |',
          '|' + '---|' * 11]
    for k in SETS:
        for q in ERRS:
            for w in WINS:
                a = R['alone'][f'{k}|{q}|{w}']
                L.append(f'| {SETNAME[k]} | {q} | {w} | ' + ' | '.join(ci(a[f]['pref_t']) for f in FAMS) + ' | '
                         + ' | '.join(ci(a[f]['use_t']) for f in FAMS) + ' |')
    L += ['', '## 表4 A-2 の赤好き・緑好き別の U_pref（好み混合の2つのデータ、6組の平均、記述）', '',
          '| データ | 誤差 | 区間 | ' + ' | '.join(f'{NAME[f]} 赤／緑' for f in FAMS) + ' |', '|' + '---|' * 7]
    for cell, v in R['liked_split'].items():
        k, q, w = cell.split('|')
        L.append(f'| {SETNAME[k]} | {q} | {w} | ' + ' | '.join(f'{np.mean(v[f]["red"]):+.4f} ／ {np.mean(v[f]["green"]):+.4f}' for f in FAMS) + ' |')
    c = R['checks']
    L += ['', f'checks: ep200 の2回目がビット単位で一致 {c["rep_all"]}（{sum(c["rep_identical"].values())}/{len(c["rep_identical"])}）；'
          f'保存済みの照合の結果と重なる {c["saved_n"]} 件のうちビット単位で一致 {c["saved_identical_n"]} 件、差の最大 {c["saved_max_abs"]}']
    txt = '\n'.join(L) + '\n'
    open(os.path.join(D, 'tables.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
