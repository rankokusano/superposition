"""
Paper round 11, E2 addition aggregation (docs/paper_extra_log.md Sec.35.8; outside the pre-registration, procedure
committed before computing). CPU, saved runs only (e2pos_run.py -> E2/pos/runs/, e2_run.py -> E2/runs/).
A  position error: per episode dev = mean over the readable frames 10-50 (t = 9..49) of |read A-2 - true A-2| (NaN if
   none readable); U^pos_pref = dev(opp) - dev(true), U^pos_use = dev(zero) - dev(true) (NaN unless both readable).
   L1-stage late 5, A-1 Red. Pair value = mean over the 5 points of the episode mean (nanmean). Comparisons as E2
   (p50 - none, p50 alone, p80 - none, p80 - sr80) on 4 sets: 6-pair t CI (across_pairs_v13.test) of the pair-value
   difference (main form) and, descriptive, of the episode-paired difference; per pair: points whose episode-paired
   bootstrap CI (eval_g5_v12.summ, NaN dropped) is > 0, 3 of 5. Readable fractions per condition / input / set.
B  red-form bias: B = L(green form) - L(red form) per episode (> 0: the red-liked P gives the smaller error), for FP
   (MSE-stage late 5, frames 1-50 and 1-100, E2 runs) and dev (L1-stage late 5, frames 10-50). Per condition and A-2's
   true preference; on the preference-mixed sets also bias = (B_red + B_green)/2 and discrimination = (B_red - B_green)/2.
C  stage-1 sign convention: v14-C (v14c s0-2, v14cr s0-1) with A-1 Red / Green at evaluation, next to v12 p50 / none of
   the same seeds; the 4 conditions with A-1 Green (position only); checks that FP does not depend on A-1's sign.
E1 (Sec.35.9, added to E1's pre-registration before E1's evaluation; only once sr50 runs exist for all 6 pairs): the
   secondary judgement U^pos_pref p50 - sr50 on the preference-mixed set, frames 10-50, L1-stage late 5, A-1 Red: holds
   if the 6-pair t CI lower bound > 0 (main form). Same forms as A (episode-paired form, per-pair points) on all 4 sets.
-> data/result/paper_extra/E2/pos/summary.json, tables.md
"""

import json
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (os.path.join(_PI, 'analyze', 'v13'), os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)
from across_pairs_v13 import test as ttest  # noqa
from eval_g5_v12 import summ  # noqa
from eval_trial_v12 import RES  # noqa

warnings.filterwarnings('ignore', message='Mean of empty slice')
POS = os.path.join(RES, 'paper_extra', 'E2', 'pos')
RUNS = os.path.join(POS, 'runs')
E2RUNS = os.path.join(RES, 'paper_extra', 'E2', 'runs')
INV = os.path.join(RES, 'v14', 'inverse')
FAMS = ['none', 'p50', 'p80', 'sr80']
NAME = {'none': '経験なし', 'p50': '経験50%', 'p80': '経験80%', 'sr80': '符号無作為（経験80%）', 'sr50': '符号無作為（経験50%）',
        'v14c': 'v14-C 経験あり', 'v14cr': 'v14-C 経験なし版'}
SEEDS = range(6)
EPS = (160, 170, 180, 190, 200)
SETS = ('pref', 'green', 'pref_a2p80', 'green_a2p80')
SETNAME = {'pref': '好み混合', 'green': '緑好み', 'pref_a2p80': 'A-2 0.8 好み混合', 'green_a2p80': 'A-2 0.8 緑好み'}
COMPS = {'p50_none': ('p50', 'none'), 'S1_p50': ('p50', None), 'S2_p80_none': ('p80', 'none'), 'S3_p80_sr80': ('p80', 'sr80')}
CANDS = ('true', 'opp', 'zero')
WPOS = slice(9, 50)
_c = {}


def ld(p):
    if p not in _c:
        with np.load(p) as z:
            _c[p] = {k: z[k] for k in z.files}
    return _c[p]


def fpos(f, s, e, k, a1='red', tag=''):
    return ld(os.path.join(RUNS, f'{f}_l1_s{s}_ep{e}_{k}' + ('_a1green' if a1 == 'green' else '') + f'{tag}.npz'))


def ffp(f, s, e, k):
    return ld(os.path.join(RUNS if f.startswith('v14') else E2RUNS, f'{f}_mse_s{s}_ep{e}_{k}.npz'))


def forms(v, liked):
    """v: {'true', 'opp', 'zero'} per episode -> (red form, green form)."""
    r = liked == 1
    return np.where(r, v['true'], v['opp']), np.where(r, v['opp'], v['true'])


def epi(f, s, e, k, q, a1='red'):
    """per-episode error per input: q = 'pos' (dev, frames 10-50) or 'FP|1-50' / 'FP|1-100'."""
    if q == 'pos':
        z = fpos(f, s, e, k, a1)
        return {c: np.nanmean(z[f'dev_{c}'][:, WPOS], 1) for c in CANDS}, z['a2_liked']
    z, n = ffp(f, s, e, k), int(q.split('-')[1])
    return {c: z[f'FP_{c}'][:, :n].mean(1) for c in CANDS}, z['a2_liked']


def qty(f, s, e, k, q, kind, a1='red'):
    v, lk = epi(f, s, e, k, q, a1)
    if kind == 'pref':
        return v['opp'] - v['true']
    if kind == 'use':
        return v['zero'] - v['true']
    r, g = forms(v, lk)
    return g - r   # kind == 'B'


def liked(f, s, k, q):
    return epi(f, s, EPS[0], k, q)[1]


def pv(f, s, k, q, kind, a1='red', mask=None):
    return float(np.mean([np.nanmean(qty(f, s, e, k, q, kind, a1)[mask] if mask is not None else qty(f, s, e, k, q, kind, a1))
                          for e in EPS]))


def comparison(a, b, k, q, kind='pref', seeds=SEEDS, a1='red'):
    dm, dp, pts = [], [], []
    for s in seeds:
        dm.append(pv(a, s, k, q, kind, a1) - (pv(b, s, k, q, kind, a1) if b else 0.0))
        per = [qty(a, s, e, k, q, kind, a1) - (qty(b, s, e, k, q, kind, a1) if b else 0.0) for e in EPS]
        dp.append(float(np.mean([np.nanmean(x) for x in per])))
        p = [summ(x) for x in per]
        ok = [x['mean'] > 0 and x['ci95'][0] > 0 for x in p]
        pts.append({'n_ok': int(sum(ok)), 'pair_ok': sum(ok) >= 3, 'points': p})
    return {'diffs': dm, 'diffs_paired': dp, 't': ttest(dm), 't_paired': ttest(dp), 'n_pos': int(sum(d > 0 for d in dm)),
            'per_pair': pts, 'pairs_ok': int(sum(x['pair_ok'] for x in pts))}


def readable(f, k, a1='red'):
    out = {}
    for c in CANDS:
        fr = [[float(np.isfinite(fpos(f, s, e, k, a1)[f'dev_{c}'][:, WPOS]).mean()) for e in EPS] for s in SEEDS]
        ep = [[float(np.isfinite(epi(f, s, e, k, 'pos', a1)[0][c]).mean()) for e in EPS] for s in SEEDS]
        out[c] = {'frames': float(np.mean(fr)), 'frames_pair_minmax': [float(np.min(np.mean(fr, 1))), float(np.max(np.mean(fr, 1)))],
                  'episodes': float(np.mean(ep))}
    for kind in ('pref', 'use'):
        u = [[float(np.isfinite(qty(f, s, e, k, 'pos', kind, a1)).mean()) for e in EPS] for s in SEEDS]
        out[f'U_{kind}_defined'] = float(np.mean(u))
    return out


def bias_block(f, k, q, seeds, a1='red'):
    lk = liked(f, seeds[0], k, q)
    r = {}
    if k.startswith('pref'):
        r['red'] = [pv(f, s, k, q, 'B', a1, lk == 1) for s in seeds]
        r['green'] = [pv(f, s, k, q, 'B', a1, lk == 0) for s in seeds]
        r['bias'] = [(x + y) / 2 for x, y in zip(r['red'], r['green'])]
        r['disc'] = [(x - y) / 2 for x, y in zip(r['red'], r['green'])]
    else:
        r['green'] = [pv(f, s, k, q, 'B', a1) for s in seeds]
    return {g: {'per_pair': v, 't': ttest(v) if len(v) > 2 else None} for g, v in r.items()}


def checks(fams_g, extra=()):
    out = {'L1_vs_e2': [], 'FP_vs_e2mse200': [], 'rep': [], 'v14_fp_a1_independent': [], 'v14_overlap': []}
    for f in FAMS + list(extra):
        for s in SEEDS:
            for k in SETS:
                for e in EPS:
                    z = fpos(f, s, e, k)
                    out['L1_vs_e2'].append(bool(z['chk_L1_vs_e2'].all()))
                    out['FP_vs_e2mse200'].append(bool(z['chk_FP_vs_e2mse200'].all()))
                    if f in fams_g:
                        out['FP_vs_e2mse200'].append(bool(fpos(f, s, e, k, 'green')['chk_FP_vs_e2mse200'].all()))
                if not os.path.exists(os.path.join(RUNS, f'{f}_l1_s{s}_ep200_{k}_rep.npz')):
                    continue
                a, b = fpos(f, s, 200, k), fpos(f, s, 200, k, tag='_rep')
                out['rep'].append(all(np.array_equal(a[x], b[x], equal_nan=True) for x in a if x.startswith('dev')))
    for f, seeds, nm in (('v14c', (0, 1, 2), 'mixedpref'), ('v14cr', (0, 1), 'randompref')):
        for s in seeds:
            for k in SETS:
                m = ffp(f, s, 200, k)
                for e in EPS:
                    for a1 in ('red', 'green'):
                        z = fpos(f, s, e, k, a1)
                        out['v14_fp_a1_independent'].append(all(np.array_equal(z[f'FP_{c}'], m[f'FP_{c}']) for c in CANDS))
                if k not in ('pref', 'green'):
                    continue
                for e in EPS:
                    for st, pre, a1 in (('l1', 'v14c', 'red'), ('l1', 'v14c', 'green'), ('mse', 'v14cmse', 'red'), ('mse', 'v14cmse', 'green')):
                        p = os.path.join(INV, f'{pre}_{nm}_{s}_ep{e}' + ('_a1green' if a1 == 'green' else '') + f'_{k}.npz')
                        if not os.path.exists(p) or (st == 'mse' and a1 == 'green'):
                            continue
                        z = ld(p)
                        n = fpos(f, s, e, k, a1) if st == 'l1' else ffp(f, s, e, k)
                        for q in ('L1', 'FP'):
                            r, g = forms({c: n[f'{q}_{c}'] for c in CANDS}, n['a2_liked'][:, None])
                            same = (np.array_equal(r, z[f'{q}_ref_red']) and np.array_equal(g, z[f'{q}_ref_green'])
                                    and np.array_equal(n[f'{q}_zero'], z[f'{q}_c0']))
                            out['v14_overlap'].append(same)
    return {k: {'n': len(v), 'all': all(v), 'n_true': int(sum(v))} for k, v in out.items()}


def main():
    have = lambda f, s, a1='red': os.path.exists(os.path.join(RUNS, f'{f}_l1_s{s}_ep200_pref' + ('_a1green' if a1 == 'green' else '') + '.npz'))
    fams_g = [f for f in FAMS if all(have(f, s, 'green') for s in SEEDS)]
    extra = [f for f in ('sr50',) if all(have(f, s) for s in SEEDS)]
    R = {'A': {'cells': {}, 'alone': {}, 'readable': {}}, 'B': {}, 'C': {}, 'fams_a1green': fams_g, 'extra_fams': extra}
    for k in SETS:
        R['A']['cells'][k] = {c: comparison(a, b, k, 'pos') for c, (a, b) in COMPS.items()}
        R['A']['alone'][k] = {}
        for f in FAMS + extra:
            v = {kind: [pv(f, s, k, 'pos', kind) for s in SEEDS] for kind in ('pref', 'use')}
            R['A']['alone'][k][f] = {kind: {'per_pair': v[kind], 't': ttest(v[kind])} for kind in v}
            R['A']['readable'][f'{k}|{f}'] = readable(f, k)
    for q in ('FP|1-50', 'FP|1-100', 'pos'):
        R['B'][q] = {k: {f: bias_block(f, k, q, list(SEEDS)) for f in FAMS + extra} for k in SETS}
    C = R['C']
    for f, seeds in (('v14c', [0, 1, 2]), ('v14cr', [0, 1])):
        C[f] = {}
        for k in SETS:
            c = {'B_FP|1-50': bias_block(f, k, 'FP|1-50', seeds), 'B_FP|1-100': bias_block(f, k, 'FP|1-100', seeds)}
            for a1 in ('red', 'green'):
                c[f'B_pos|{a1}'] = bias_block(f, k, 'pos', seeds, a1)
                c[f'Upos|{a1}'] = {kind: [pv(f, s, k, 'pos', kind, a1) for s in seeds] for kind in ('pref', 'use')}
            c['U_FP|1-50'] = {kind: [pv(f, s, k, 'FP|1-50', kind) for s in seeds] for kind in ('pref', 'use')}
            c['readable|red'] = {cc: float(np.mean([[np.isfinite(epi(f, s, e, k, 'pos')[0][cc]).mean() for e in EPS] for s in seeds]))
                                 for cc in CANDS}
            C[f][k] = c
    C['v12ref'] = {}
    for f, seeds in (('p50', [0, 1, 2]), ('none', [0, 1])):
        C['v12ref'][f] = {k: {'B_FP|1-50': bias_block(f, k, 'FP|1-50', seeds), 'B_FP|1-100': bias_block(f, k, 'FP|1-100', seeds),
                              'B_pos|red': bias_block(f, k, 'pos', seeds),
                              'B_pos|green': bias_block(f, k, 'pos', seeds, 'green') if f in fams_g else None,
                              'U_FP|1-50': {kind: [pv(f, s, k, 'FP|1-50', kind) for s in seeds] for kind in ('pref', 'use')},
                              'Upos|red': {kind: [pv(f, s, k, 'pos', kind) for s in seeds] for kind in ('pref', 'use')}}
                          for k in SETS}
    C['v14c_minus_p50'] = {}
    for k in ('pref', 'pref_a2p80'):
        d = {}
        for q in ('B_FP|1-50', 'B_pos|red'):
            for g in ('bias', 'disc'):
                x = [a - b for a, b in zip(C['v14c'][k][q][g]['per_pair'], C['v12ref']['p50'][k][q][g]['per_pair'])]
                d[f'{q}|{g}'] = {'per_pair': x, 't': ttest(x)}
        C['v14c_minus_p50'][k] = d
    C['a1green'] = {}
    for f in fams_g:
        C['a1green'][f] = {}
        for k in SETS:
            bg, br = bias_block(f, k, 'pos', list(SEEDS), 'green'), R['B']['pos'][k][f]
            ch = {g: [a - b for a, b in zip(bg[g]['per_pair'], br[g]['per_pair'])] for g in bg}
            C['a1green'][f][k] = {'B_pos': bg, 'change_green_minus_red': {g: {'per_pair': v, 't': ttest(v)} for g, v in ch.items()},
                                  'Upos': {kind: [pv(f, s, k, 'pos', kind, 'green') for s in SEEDS] for kind in ('pref', 'use')}}
    if 'sr50' in extra:   # Sec.35.9: E1 secondary judgement (p50 - sr50, preference-mixed set, holds if lower > 0)
        R['E1'] = {k: comparison('p50', 'sr50', k, 'pos') for k in SETS}
        R['E1']['judgement'] = {'set': 'pref', 'holds': bool(R['E1']['pref']['t']['ci95'][0] > 0),
                                'holds_paired_form': bool(R['E1']['pref']['t_paired']['ci95'][0] > 0)}
    R['checks'] = checks(fams_g, extra)
    with open(os.path.join(POS, 'summary.json'), 'w') as fh:
        json.dump(R, fh, indent=1)
    write_tables(R)


def ci(t, nd=4):
    return '—' if t is None else f'{t["mean"]:+.{nd}f} [{t["ci95"][0]:+.{nd}f}, {t["ci95"][1]:+.{nd}f}]'


def f4(x):
    return f'{x:+.4f}'


def write_tables(R):
    A, B, C = R['A'], R['B'], R['C']
    fams = FAMS + R['extra_fams']
    L = ['## 表A1 位置のずれの U（好み混合、フレーム 10〜50、L1 期間の最終5点の平均、A-1 赤好き）', '',
         '| 条件 | 量 | ' + ' | '.join(f'組{s + 1}' for s in SEEDS) + ' | 6組の平均 [95% CI] |', '|' + '---|' * 9]
    for f in fams:
        for kind in ('pref', 'use'):
            v = A['alone']['pref'][f][kind]
            L.append(f'| {NAME[f]} | U^pos_{kind} | ' + ' | '.join(f4(x) for x in v['per_pair']) + f' | {ci(v["t"])} |')
    for c, (a, b) in COMPS.items():
        X = A['cells']['pref'][c]
        lab = f'{NAME[a]} − {NAME[b]}' if b else f'{NAME[a]} 単独'
        L.append(f'| {lab} | U^pos_pref の差 | ' + ' | '.join(f4(x) for x in X['diffs']) + f' | {ci(X["t"])} |')
        L.append(f'| {lab} | 組ごとの点（5点中） | ' + ' | '.join(str(p['n_ok']) for p in X['per_pair']) + ' | |')
    L += ['', '## 表A2 U^pos_pref の比較（4 データ、フレーム 10〜50。6組の t の 95% CI、自由度5。記述）', '',
          '| データ | 比較 | 主の形 [95% CI] | エピソードを対にした形 [95% CI] | 正の組 | 組ごとの点で正の組 | 符号検定 p |', '|---|---|---|---|---|---|---|']
    for k in SETS:
        for c, (a, b) in COMPS.items():
            X = A['cells'][k][c]
            lab = f'{NAME[a]} − {NAME[b]}' if b else f'{NAME[a]} 単独'
            L.append(f'| {SETNAME[k]} | {lab} | {ci(X["t"])} | {ci(X["t_paired"])} | {X["n_pos"]}/6 | {X["pairs_ok"]}/6 | {X["t"]["sign_p"]:.3f} |')
    L += ['', '## 表A3 各条件の U^pos（6組の平均 [95% CI]）', '', '| データ | 条件 | U^pos_pref | U^pos_use |', '|---|---|---|---|']
    for k in SETS:
        for f in fams:
            v = A['alone'][k][f]
            L.append(f'| {SETNAME[k]} | {NAME[f]} | {ci(v["pref"]["t"])} | {ci(v["use"]["t"])} |')
    L += ['', '## 表A4 読めた割合（フレーム 10〜50。6組×5点の平均。フレームの割合の組の範囲は［最小, 最大］）', '',
          '| データ | 条件 | 正：フレーム | 反：フレーム | 零：フレーム | 正：エピソード | 反：エピソード | 零：エピソード | U_pref が作れた | U_use が作れた |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for k in SETS:
        for f in fams:
            r = A['readable'][f'{k}|{f}']
            L.append(f'| {SETNAME[k]} | {NAME[f]} | ' + ' | '.join(
                f'{r[c]["frames"]:.3f} [{r[c]["frames_pair_minmax"][0]:.3f}, {r[c]["frames_pair_minmax"][1]:.3f}]' for c in CANDS) + ' | '
                + ' | '.join(f'{r[c]["episodes"]:.3f}' for c in CANDS) + f' | {r["U_pref_defined"]:.3f} | {r["U_use_defined"]:.3f} |')
    qn = {'FP|1-50': '他者特徴の誤差（FP）、フレーム 1〜50', 'FP|1-100': 'FP、フレーム 1〜100', 'pos': '位置のずれ、フレーム 10〜50'}
    gn = {'red': 'A-2 赤好き', 'green': 'A-2 緑好き', 'bias': '偏り (赤好き+緑好き)/2', 'disc': '見分け (赤好き−緑好き)/2'}
    L += ['', '## 表B 向きの偏り B = L(緑の形) − L(赤の形)（正：「赤が好き」とした P の方が誤差が小さい。6組の平均 [95% CI]、A-1 赤好き）', '']
    for q in B:
        L += [f'### {qn[q]}', '', '| データ | 本当の好み | ' + ' | '.join(NAME[f] for f in fams) + ' |', '|' + '---|' * (len(fams) + 2)]
        for k in SETS:
            for g in B[q][k][fams[0]]:
                L.append(f'| {SETNAME[k]} | {gn[g]} | ' + ' | '.join(ci(B[q][k][f][g]['t']) for f in fams) + ' |')
        L.append('')
    L += ['## 表C1 v14-C（段階I で A-1 の好きな色をエピソードごとに変えた）と v12（同じ組）の偏りと見分け（組ごとの値）', '',
          '| データ | 量 | モデル | A-1（評価） | 組1 | 組2 | 組3 |', '|---|---|---|---|---|---|---|']
    for k in ('pref', 'pref_a2p80'):
        for q, a1s in (('B_FP|1-50', [None]), ('B_FP|1-100', [None]), ('B_pos', ['red', 'green'])):
            for g in ('bias', 'disc'):
                for f, src in (('p50', C['v12ref']['p50'][k]), ('v14c', C['v14c'][k]), ('none', C['v12ref']['none'][k]), ('v14cr', C['v14cr'][k])):
                    for a1 in a1s:
                        key = q if a1 is None else f'{q}|{a1}'
                        if src.get(key) is None:
                            continue
                        v = src[key][g]['per_pair']
                        L.append(f'| {SETNAME[k]} | {q.replace("B_", "B ")} {gn[g]} | {NAME[f]} | {a1 or "（FP は A-1 に依らない）"} | '
                                 + ' | '.join(f4(x) for x in v) + ' |' * (4 - len(v)))
    L += ['', '緑好きだけのデータ（B の値。正：赤の形が有利）：', '', '| データ | 量 | モデル | A-1（評価） | 組1 | 組2 | 組3 |', '|---|---|---|---|---|---|---|']
    for k in ('green', 'green_a2p80'):
        for q, a1s in (('B_FP|1-50', [None]), ('B_pos', ['red', 'green'])):
            for f, src in (('p50', C['v12ref']['p50'][k]), ('v14c', C['v14c'][k]), ('none', C['v12ref']['none'][k]), ('v14cr', C['v14cr'][k])):
                for a1 in a1s:
                    key = q if a1 is None else f'{q}|{a1}'
                    if src.get(key) is None:
                        continue
                    v = src[key]['green']['per_pair']
                    L.append(f'| {SETNAME[k]} | {q.replace("B_", "B ")} | {NAME[f]} | {a1 or "—"} | ' + ' | '.join(f4(x) for x in v) + ' |' * (4 - len(v)))
    L += ['', 'v14-C − v12 経験50%（同じ組1〜3、t の 95% CI、自由度2、記述）：', '', '| データ | 量 | 差 [95% CI] | 組ごと |', '|---|---|---|---|']
    for k, d in C['v14c_minus_p50'].items():
        for q, v in d.items():
            L.append(f'| {SETNAME[k]} | {q} | {ci(v["t"])} | ' + ', '.join(f4(x) for x in v['per_pair']) + ' |')
    L += ['', 'v14-C の U（組ごと。FP は MSE 期間・1〜50、位置は L1 期間・10〜50）：', '', '| データ | モデル | 量 | 組1 | 組2 | 組3 |', '|---|---|---|---|---|---|']
    for k in SETS:
        for f in ('v14c', 'v14cr'):
            for q in ('U_FP|1-50', 'Upos|red', 'Upos|green'):
                for kind in ('pref', 'use'):
                    v = C[f][k][q][kind]
                    L.append(f'| {SETNAME[k]} | {NAME[f]} | {q} {kind} | ' + ' | '.join(f4(x) for x in v) + ' |' * (4 - len(v)))
    if C['a1green']:
        L += ['', '## 表C2 4 条件で A-1 を緑好きにしたときの位置の偏り（6組の平均 [95% CI]。学習の分布の外）', '',
              '| データ | 本当の好み | 条件 | B（A-1 赤） | B（A-1 緑） | 緑 − 赤 |', '|---|---|---|---|---|---|']
        for k in SETS:
            for f in C['a1green']:
                X = C['a1green'][f][k]
                for g in X['B_pos']:
                    L.append(f'| {SETNAME[k]} | {gn[g]} | {NAME[f]} | {ci(B["pos"][k][f][g]["t"])} | {ci(X["B_pos"][g]["t"])} | '
                             f'{ci(X["change_green_minus_red"][g]["t"])} |')
    if 'E1' in R:
        X, J = R['E1'], R['E1']['judgement']
        L += ['', '## 表E1 E1 の副の判定（§35.9）：U^pos_pref の 経験50% − 符号無作為（経験50%）（フレーム 10〜50、L1 期間の最終5点、A-1 赤好き）', '',
              f'判定（好み混合、主の形、6組の t の CI の下限 > 0）：{"成り立つ" if J["holds"] else "成り立たない"}'
              f'（エピソードを対にした形：{"下限 > 0" if J["holds_paired_form"] else "0 を含むか下限 ≤ 0"}）', '',
              '| データ | ' + ' | '.join(f'組{s + 1}' for s in SEEDS) + ' | 主の形 [95% CI] | エピソードを対にした形 [95% CI] | 正の組 | 組ごとの点で正の組 |',
              '|' + '---|' * 11]
        for k in SETS:
            x = X[k]
            L.append(f'| {SETNAME[k]} | ' + ' | '.join(f4(d) for d in x['diffs']) + f' | {ci(x["t"])} | {ci(x["t_paired"])} | '
                     f'{x["n_pos"]}/6 | {x["pairs_ok"]}/6 |')
    L += ['', f'checks: {json.dumps(R["checks"], ensure_ascii=False)}']
    txt = '\n'.join(L) + '\n'
    open(os.path.join(POS, 'tables.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
