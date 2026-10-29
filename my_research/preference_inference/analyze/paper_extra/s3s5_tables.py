"""
Paper round 6 (docs/paper_extra_log.md Sec.34.8): tables A / B and the exclusion table as markdown, from
C_signrand/s3s5/summary.json (s3s5_aggregate.py) only. No new judgement: the per-pair points, the 6-pair t CIs and the
verdicts are read from the summary. The t CIs of tracking, BA and the (hold + flip)/2 form of g are descriptive
(same across_pairs_v13.test; not in the pre-registration's judgement).
-> data/result/paper_extra/C_signrand/s3s5/tables.md
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v13'))
from across_pairs_v13 import test as ttest  # noqa

D = os.path.join(_PI, 'data', 'result', 'paper_extra', 'C_signrand', 's3s5')
S = json.load(open(os.path.join(D, 'summary.json')))
V, P = S['values'], S['per_pair']
SEEDS = range(6)
NAME = {'signrand': '符号無作為', 'mixed80': '経験80%', 'random': '経験なし'}

# (key in values, label, key of the per-pair judgement vs 80% or None, key of across6 or None)
COMMON = [('dP', '$d_P$（V3）', 'dP_vs80', 'dP'),
          ('G5b', 'G5b（V1）', 'G5b_vs80', 'G5b'),
          ('signfollow_green', '符号に従う分 ①（緑好み，V1）', 'signfollow_green_vs80', 'signfollow_green'),
          ('signfollow_pref', '符号に従う分 ②（好み混合，V1）', 'signfollow_pref_vs80', 'signfollow_pref'),
          ('G3hold', '$g$ の残り（G3 の保つ）', 'G3hold_vs80', 'G3hold'),
          ('g_rest_holdflip', '$g$ の残り（記述：(保つ+反転)/2）', None, None),
          ('G1_visible', '自己位置の追跡（G1，見える区間）', None, None),
          ('BA_L1_l1late5', '照合の BA（自己視界基準，L1 期間）', None, None),
          ('BA_FP_mselate5', '照合の BA（他者特徴基準，MSE 期間）', None, None),
          ('read_dP', '読めた割合（$d_P$）', None, None),
          ('read_G5b', '読めた割合（G5b）', None, None),
          ('read_green', '読めた割合（①）', None, None),
          ('read_pref', '読めた割合（②）', None, None)]
B4 = [('B4_G5b', 'G5b（B4-V1）', 'B4_G5b_vs80', 'B4_G5b'),
      ('B4_signfollow_green', '符号に従う分 ①（B4-V1）', 'B4_signfollow_green_vs80', 'B4_signfollow_green'),
      ('B4_signfollow_pref', '符号に従う分 ②（B4-V1）', 'B4_signfollow_pref_vs80', 'B4_signfollow_pref'),
      ('B4_track_visible', '追跡（見える区間）', None, None),
      ('B4_track_hidden', '追跡（隠れた区間）', None, None),
      ('B4_read_G5b', '読めた割合（G5b）', None, None),
      ('B4_read_green', '読めた割合（①）', None, None),
      ('B4_read_pref', '読めた割合（②）', None, None)]


def mean(x):
    return float(np.mean(x)) if isinstance(x, list) else float(x)


def val(s, c, q):
    v = V[f's{s}'][c].get(q)
    return None if v is None else mean(v)


def f(x, sign=True):
    return '—' if x is None else (f'{x:+.3f}' if sign else f'{x:.3f}')


def table_a(rows):
    out = []
    hdr = '| 量 | 行 | ' + ' | '.join(f'組{s + 1}（s{s}）' for s in SEEDS) + ' |'
    out += [hdr, '|' + '---|' * (len(SEEDS) + 2)]
    for q, lab, jk, ak in rows:
        sg = not q.startswith(('read', 'B4_read', 'BA', 'G1', 'B4_track'))
        for c in ('signrand', 'mixed80', 'random'):
            out.append(f'| {lab} | {NAME[c]} | ' + ' | '.join(f(val(s, c, q), sg) for s in SEEDS) + ' |')
        d = [None if val(s, 'signrand', q) is None or val(s, 'mixed80', q) is None
             else val(s, 'signrand', q) - val(s, 'mixed80', q) for s in SEEDS]
        out.append(f'| {lab} | 差（符号無作為 − 80%） | ' + ' | '.join(f(x) for x in d) + ' |')
        if ak:
            out.append(f'| {lab} | 差（エピソードを対にした形） | '
                       + ' | '.join(f(x) for x in S['across6'][ak]['diff_paired']) + ' |')
        if jk:
            pts = [P[f's{s}'][jk]['n_ok'] for s in SEEDS]
            out.append(f'| {lab} | 判定 vs 80%（5点中） | '
                       + ' | '.join(f'{n}/5 {"合格" if n >= 3 else "**不合格**"}' for n in pts) + ' |')
        if q == 'G3hold':
            pts = [P[f's{s}']['G3hold_vsnone']['n_ok'] for s in SEEDS]
            out.append(f'| {lab} | 判定 符号無作為 > 経験なし（P1，5点中） | '
                       + ' | '.join(f'{n}/5 {"合格" if n >= 3 else "**不合格**"}' for n in pts) + ' |')
    return out


def ci(t):
    return f'{t["mean"]:+.3f} [{t["ci95"][0]:+.3f}, {t["ci95"][1]:+.3f}]'


def table_b():
    out = ['| 量 | 役割 | 6組：差の平均 [95% CI]（主の形） | 6組：対にした形 | 負の組（主／対） | 符号検定 p | '
           '組ごとの判定に合格（6組中） | 組1〜3：差の平均 [95% CI]（自由度2，主の形） | 組1〜3 の符号検定 p |',
           '|---|---|---|---|---|---|---|---|---|']
    role = {'dP': '主', 'G5b': '主'}
    lab = dict((a, b) for a, b, *_ in COMMON + B4)
    for q in ('dP', 'G5b', 'signfollow_green', 'signfollow_pref', 'B4_G5b', 'B4_signfollow_green', 'B4_signfollow_pref', 'G3hold'):
        A = S['across6'][q]
        jk = f'{q}_vs80'
        n6 = S['judge_counts'][jk]['all6']
        out.append(f'| {lab[q]}{"（共通）" if not q.startswith("B4") else "（B4）"} | {role.get(q, "副" if q != "G3hold" else "記述")} | '
                   f'{ci(A["all6_means"])} | {ci(A["all6_paired"])} | {A["n_neg_means"]}/6 ／ {A["n_neg_paired"]}/6 | '
                   f'{A["all6_means"]["sign_p"]:.3f} | {n6}/6 | {ci(A["main3_means"])} | {A["main3_means"]["sign_p"]:.2f} |')
    out += ['', '記述（事前登録の判定に入れない。同じ $t$，自由度5，主の形）：', '',
            '| 量 | 6組：差の平均 [95% CI] | 負の組 | 符号検定 p |', '|---|---|---|---|']
    for q in ('g_rest_holdflip', 'G1_visible', 'BA_L1_l1late5', 'BA_FP_mselate5', 'B4_track_visible', 'B4_track_hidden'):
        d = [val(s, 'signrand', q) - val(s, 'mixed80', q) for s in SEEDS]
        t = ttest(d)
        out.append(f'| {lab[q]} | {ci(t)} | {sum(x < 0 for x in d)}/6 | {t["sign_p"]:.3f} |')
    return out


def table_delta():
    out = ['| 真値条件 Δ（L1 期間 ep200） | 行 | ' + ' | '.join(f'組{s + 1}' for s in SEEDS) + ' |',
           '|' + '---|' * (len(SEEDS) + 2)]
    for k, lab in (('delta_green', '緑好み'), ('delta_pref', '好み混合')):
        for c in ('signrand', 'mixed80', 'random'):
            out.append(f'| {lab} | {NAME[c]} | ' + ' | '.join(f(val(s, c, k)) for s in SEEDS) + ' |')
        for nm, t in (('vs80', '符号無作為 − 80% < 0（V2）'), ('vsnone', '符号無作為 − 経験なし > 0（P1）')):
            out.append(f'| {lab} | {t} | ' + ' | '.join(
                f'{P[f"s{s}"][f"{k}_{nm}"]["diff"]["mean"]:+.3f} {"成立" if P[f"s{s}"][f"{k}_{nm}"]["ok"] else "不成立"}'
                for s in SEEDS) + ' |')
    return out


def table_excl():
    out = ['| 量 | 扱い | 組ごとの点（組1〜6，5点中） | 合格（組4〜6／6組） | 6組：主の形 [95% CI] | 6組：対にした形 [95% CI] | 負の組 |',
           '|---|---|---|---|---|---|---|']
    nm = {'orig': 'そのまま', 'a': '(a) 組の全条件で読めた', 'a3': '(a3) なし・80%・符号無作為で読めた',
          'b1': '(b1) 読めない→両方0', 'b2': '(b2) 読めない→80%だけ0'}
    for q, lab in (('G5b', 'G5b'), ('signfollow_green', '①'), ('signfollow_pref', '②')):
        for v in ('orig', 'a', 'a3', 'b1', 'b2'):
            X = S['exclusion'][f'{q}|{v}']
            pts = ' / '.join(str(X['per_pair'][f's{s}']['n_ok']) for s in SEEDS)
            neg = f'{sum(d < 0 for d in X["diff_means"])}/6 ／ {sum(d < 0 for d in X["diff_paired"])}/6'
            out.append(f'| {lab} | {nm[v]} | {pts} | {X["pairs_ok_new3"]}/3 ／ {X["pairs_ok_all6"]}/6 | '
                       f'{ci(X["t_means"])} | {ci(X["t_paired"])} | {neg} |')
    return out


def main():
    L = ['## 表A-1 共通の評価データ（最終5点の平均）', ''] + table_a(COMMON)
    L += ['', '## 表A-2 追跡のできる評価データ（B4）（最終5点の平均）', ''] + table_a(B4)
    L += ['', '## 表A-3 真値条件 Δ（V2・P1）', ''] + table_delta()
    L += ['', '## 表B 6組をまたいだ差（符号無作為 − 経験80%）', ''] + table_b()
    L += ['', '## 除外の扱い（事前登録外）', ''] + table_excl()
    L += ['', f'checks: {json.dumps(S["checks"])}', f'pairs 1-3 equal saved summaries: {S["pairs123_equal_saved_summaries"]}',
          f'verdict pairs 4-6: {json.dumps(S["verdict_new3"])}',
          f'main 1 (6 pairs): main form {S["main1_6pairs_main_form"]} | paired form {S["main1_6pairs_paired_form"]}']
    txt = '\n'.join(L) + '\n'
    open(os.path.join(D, 'tables.md'), 'w').write(txt)
    print(txt)


if __name__ == '__main__':
    main()
