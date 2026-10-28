"""
Paper round 4: markdown tables from data/result/paper_extra/R4/summary.json (r4_exclusion.py) and, for side-by-side
reading only, the B4 read fractions (B4/summary.json and the B4 per-episode files). Prints to stdout.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(_PI, 'data', 'result')
R = json.load(open(os.path.join(RES, 'paper_extra', 'R4', 'summary.json')))
EPS = (160, 170, 180, 190, 200)
LAB = {'none': '経験なし', '50': '経験50%', '80': '経験80%', 'NS': '非共有', 'NSnone': '非共有・経験なし', 'signrand': '符号無作為',
       'pzero': '零入力', 'copy': '全部写す', 'pcols': '列だけ写す'}
QL = {'G5b': 'G5b', 'green': '①緑好み', 'pref': '②好み混合'}
CL = {'80_minus_none': '経験80% − 経験なし', '50_minus_none': '経験50% − 経験なし', '80_minus_50': '経験80% − 経験50%',
      'signrand_minus_80': '符号無作為 − 経験80%', 'pzero_minus_80': '零入力 − 経験80%', 'signrand_minus_none': '符号無作為 − 経験なし',
      'NS_gate': '非共有（判定：非共有 − 非共有・経験なし）', 'NS_minus_80': '非共有 − 経験80%', 'B1_copy': '写し B1（全部）',
      'B3_pcols': '写し B3（列だけ）'}


def rng(v, f='{:.3f}'):
    return f.format(min(v)) if min(v) == max(v) else f'{f.format(min(v))}〜{f.format(max(v))}'


def t1():
    print('### 作業1：読めた割合（最終5点の範囲）\n')
    print('| 組 | モデル | G5b エピソード | G5b 対照の試行（json） | ① 両方 | ① freeze / 反転 | ② 両方 | ② freeze / 反転 |')
    print('|---|---|---|---|---|---|---|---|')
    keys = sorted({k.split('|')[0] for k in R['readable']}, key=list(LAB).index)
    for s in range(6):
        for m in keys:
            ds = [R['readable'].get(f'{m}|s{s}|ep{e}') for e in EPS]
            if ds[0] is None:
                continue
            c = lambda k: rng([d[k] for d in ds])  # noqa
            print(f'| 組{s + 1} | {LAB[m]} | {c("G5b_episode")} | {c("G5b_frac_read_ctrl")} | {c("green_both")} | '
                  f'{c("green_freeze")} / {c("green_flip")} | {c("pref_both")} | {c("pref_freeze")} / {c("pref_flip")} |')


def t1_b4():
    sys.path.insert(0, _HERE)
    import b4_aggregate as ba
    print('\n### 参考：B4 の評価データ（自分の学習データと同じ作り）の読めた割合（最終5点の範囲）\n')
    print('| 組 | モデル | G5b エピソード | G5b 対照の試行 | ① 両方 | ② 両方 |')
    print('|---|---|---|---|---|---|')
    B = json.load(open(os.path.join(RES, 'paper_extra', 'B4', 'summary.json')))['per_pair']
    for s in range(3):
        for c, m in (('mixed80', '80'), ('signrand', 'signrand'), ('random', 'none')):
            g = [float(np.isfinite(ba.g5b(c, s, e)).mean()) for e in EPS]
            sg = [float(np.isfinite(ba.sf(c, s, e, 'green')).mean()) for e in EPS]
            sp = [float(np.isfinite(ba.sf(c, s, e, 'pref')).mean()) for e in EPS]
            print(f'| 組{s + 1} | {LAB[m]} | {rng(g)} | {rng(B[f"s{s}"]["values"][c]["frac_read_ctrl"])} | {rng(sg)} | {rng(sp)} |')


def t2():
    print('\n### 作業2：両方で読めたエピソードの数（300 のうち、最終5点の範囲）\n')
    print('| 比較 | 組 | G5b | ① | ② |')
    print('|---|---|---|---|---|')
    cc = R['common_counts']
    for name in CL:
        for s in range(6):
            if f'{name}|G5b|s{s}' not in cc:
                continue
            print(f'| {CL[name]} | 組{s + 1} | ' + ' | '.join(rng(cc[f"{name}|{q}|s{s}"], '{}') for q in QL) + ' |')
    print('\n**組の中のすべての条件で読めた数**（組1〜3：9 モデル、組4〜6：3 モデル；［ ］は経験なし・50%・80% の 3 モデル）\n')
    print('| 組 | G5b | ① | ② |')
    print('|---|---|---|---|')
    for s in range(6):
        print(f'| 組{s + 1} | ' + ' | '.join(f'{rng(cc[f"ALL|{q}|s{s}"], "{}")} ［{rng(cc[f"ALL3|{q}|s{s}"], "{}")}］' for q in QL) + ' |')


def t3():
    print('\n### 作業3：読めた割合の条件差（同じエピソードを対にした差、点ごとの平均の範囲；CI が 0 を含まない点の数；|差| ≥ 0.05 の点の数）\n')
    print('| 差 | 組 | G5b | ① | ② |')
    print('|---|---|---|---|---|')
    seen = []
    for k in R['readable_diff']:
        base = k.split('|')[0]
        if base not in seen:
            seen.append(base)
    for base in seen:
        a, b = base.split('_minus_')
        for s in range(6):
            if f'{base}|G5b|s{s}' not in R['readable_diff']:
                continue
            cells = []
            for q in QL:
                pts = R['readable_diff'][f'{base}|{q}|s{s}']
                mv = [p['mean'] for p in pts]
                nci = sum(p['ci95'][0] > 0 or p['ci95'][1] < 0 for p in pts)
                big = sum(abs(x) >= 0.05 for x in mv)
                cells.append(f'{rng(mv, "{:+.3f}")}（{nci}/5；{big}）')
            print(f'| {LAB[a]} − {LAB[b]} | 組{s + 1} | ' + ' | '.join(cells) + ' |')


def t3_props():
    P = R['episode_props']
    print('\n### 作業3：読めないエピソード（5 点のどれかで欠け）の性質（読めない / 読める の平均）\n')
    for q in QL:
        d = P[f'_data|{q}']
        print(f'- {QL[q]} のデータ全体：A-2 から最も近い角まで {d["A2_to_corner_mean"]:.2f}、A-1 と A-2 の距離 {d["A1_A2_dist_mean"]:.2f}、'
              f'元の配置の割合 {d["layout0_frac"]:.3f}' + (f'、A-2 の t = 20〜29 の最大の動き {d["A2_max_move_t20_29"]:.3g}' if d['A2_max_move_t20_29'] is not None else ''))
    print('\n| 量 | 組 | モデル | 読めない数（どれか / 5 点とも） | A-2 → 角 | A-1 と A-2 の距離 | 元の配置 | A-2 が赤好み |')
    print('|---|---|---|---|---|---|---|---|')
    for q in QL:
        for s in range(6):
            for m in LAB:
                r = P.get(f'{m}|{q}|s{s}')
                if r is None:
                    continue
                lk = f'{r["a2_likes_red_unreadable"]:.2f} / {r["a2_likes_red_readable"]:.2f}' if 'a2_likes_red_unreadable' in r else '—'
                print(f'| {QL[q]} | 組{s + 1} | {LAB[m]} | {r["n_unreadable_any"]} / {r["n_unreadable_all5"]} | '
                      f'{r["A2_to_corner_unreadable"]:.2f} / {r["A2_to_corner_readable"]:.2f} | {r["A1_A2_dist_unreadable"]:.2f} / {r["A1_A2_dist_readable"]:.2f} | '
                      f'{r["layout0_unreadable"]:.2f} / {r["layout0_readable"]:.2f} | {lk} |')


def t4():
    C = R['comparisons']
    print('\n### 作業4：判定（成立した組の数。組の中の点の数は 組1/組2/…）\n')
    print('| 比較 | 量 | 元 | (a) 共通集合 | (b1) 両側 0 | (b2) 大きい側だけ 0 |')
    print('|---|---|---|---|---|---|')
    changed = []
    for name in CL:
        for q in QL:
            cells = []
            base = C[f'{name}|{q}|orig']
            for v in ('orig', 'a', 'b1', 'b2'):
                r = C[f'{name}|{q}|{v}']
                pts = '/'.join(str(r['per_pair'][k]['n_ok']) for k in sorted(r['per_pair']))
                tot = f'{r["pairs_ok_main3"]}/3' + (f'（6 組 {r["pairs_ok_all"]}/6）' if r['n_pairs'] == 6 else '')
                cells.append(f'{tot}［{pts}］')
                pc = [k for k in r['per_pair'] if r['per_pair'][k]['pair_ok'] != base['per_pair'][k]['pair_ok']]
                need = 2
                verdict = lambda x: x['pairs_ok_main3'] >= need  # noqa
                if pc or verdict(r) != verdict(base):
                    changed.append((CL[name], QL[q], v, pc, f'{base["pairs_ok_main3"]}/3→{r["pairs_ok_main3"]}/3',
                                    f'{base["pairs_ok_all"]}/{base["n_pairs"]}→{r["pairs_ok_all"]}/{r["n_pairs"]}'))
            print(f'| {CL[name]} | {QL[q]} | ' + ' | '.join(cells) + ' |')
    print('\n**組の判定または全体の判定が元と変わったもの**\n')
    for c in changed:
        print(f'- {c[0]}・{c[1]}・{c[2]}：変わった組 {", ".join(f"組{int(k[1:]) + 1}" for k in c[3]) or "なし"}、本番3組 {c[4]}、全組 {c[5]}')
    A = R['across_pairs']
    print('\n### 作業4：組をまたいだ t の 95% CI（6 組、各組の最終5点の平均の差）\n')
    print('| 量 | 比較 | 元 | (a) 共通集合 | (a3) 3 モデル | (b1) 両側 0 | (b2) 大きい側だけ 0 |')
    print('|---|---|---|---|---|---|---|')
    for q in QL:
        for ab in ('80_minus_50', '80_minus_none', '50_minus_none'):
            cells = []
            for v in ('orig', 'a', 'a3', 'b1', 'b2'):
                t = A[f'{q}|{ab}|{v}']['all6']
                ex = t['ci95'][0] > 0 or t['ci95'][1] < 0
                cells.append(f'{t["mean"]:+.3f} [{t["ci95"][0]:+.3f}, {t["ci95"][1]:+.3f}]{"" if ex else " *"}')
            print(f'| {QL[q]} | {CL[ab]} | ' + ' | '.join(cells) + ' |')
    print('\n（* は CI が 0 を含む）')


if __name__ == '__main__':
    t1(); t1_b4(); t2(); t3(); t3_props(); t4()
