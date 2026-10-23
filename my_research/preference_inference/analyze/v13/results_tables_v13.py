"""
v13 numbers log (docs/v13_results_log.md): stage-1 tables generated from the saved aggregates
(data/result/v13/main/{s1eval,g5,subsets,track}, data/result/v13/v12ref/track, data/result/v13/gate61).
Pure python (no numpy). Prints markdown to stdout.
"""

import json
import os

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(_PI, 'data', 'result')
M = os.path.join(R, 'v13', 'main')
EPS = (160, 170, 180, 190, 200)


def ci(x):
    return f"{x['mean']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def main():
    L = ['# v13 数値ログ', '',
         '- 経緯・事前登録・判断は `docs/v13_experiment_log.md`。ここは数値の一次記録（`analyze/v13/results_tables_v13.py` で保存した集計から生成）。',
         '- 共通の test＝`v11_s1_random` の test 300。経験なし＝v12 の `v12_s1_random_*`（再利用、ログ §1）、経験あり＝v13 の `v13_s1_mixed_*`（80%）。', '']
    # 1 data
    g61 = json.load(open(os.path.join(R, 'v13', 'gate61', 'gate61_v13.json')))
    L += ['## 1. データ（ログ §2）', '', '| 項目 | 経験なし | 50%（v11/v12） | 80%（v13） |', '|---|---|---|---|']
    it = g61['item4_5']
    for nm, f in (('赤から 5 以内（全フレーム）', lambda r: f"{r['near_red5_all']:.3f}"),
                  ('同、t=50〜100', lambda r: f"{r['near_red5_t50_100']:.3f}"),
                  ('起きている一歩：規則 / 使えずランダム / ランダム', lambda r: f"{r['source_frac_of_awake']['rule']:.3f} / {r['source_frac_of_awake']['rule_fallback']:.3f} / {r['source_frac_of_awake']['random']:.3f}"),
                  ('最初に 5 以内に入る時刻の中央値', lambda r: f"{r['first_within5_median']:.0f}"),
                  ('快の正の部分の和の平均', lambda r: f"{r['pleasure']['liked_bins_mean_positive_part']:.3f}")):
        L.append(f'| {nm} | ' + ' | '.join(f(it[c]) for c in ('random', 'mixed50', 'mixed80')) + ' |')
    # 2 G1-G3 per checkpoint
    gs = json.load(open(os.path.join(M, 's1eval', 'gate_s1.json')))
    L += ['', '## 2. G1〜G3（mse 段、ログ §11）', '', f"- 関門：{gs['gate']}", '',
          '| 組 | ep | G1 見える一歩 R² 経験なし / 経験あり | M1(c) 経験あり / 組の差 | M1(d) 経験あり / 組の差 | 色の基準 経験あり | G3 g 経験あり / 組の差 |',
          '|---|---|---|---|---|---|---|']
    for s in '012':
        for r in gs['per_seed'][s]['rows']:
            v = r['values']
            L.append(f"| s{s} | {r['ep']} | {v['track']['random']['visible']:.3f} / {v['track']['mixed']['visible']:.3f} | "
                     f"{v['M1c_mixed']:+.4f} / {v['M1c_paired']:+.4f} | {v['M1d_mixed']:+.4f} / {v['M1d_paired']:+.4f} | "
                     f"{v['colour_mixed']:+.4f} | {v['G3_mixed']:+.3f} / {v['G3_paired']:+.3f} |")
    # 3 G5
    g5 = json.load(open(os.path.join(M, 'g5', 'gate_g5.json')))
    L += ['', '## 3. G5a・G5b（l1 段、ログ §11）', '', f"- 関門：{g5['gate']}、副（B1）：{g5['secondary_bg']}", '',
          '| 組 | ep | G5a ① 経験あり | G5a ① 組の差 | G5a ② 経験あり | G5a ② 組の差 | G5b 経験あり / 組の差 | 判定 G5a / G5b |',
          '|---|---|---|---|---|---|---|---|']
    for s in '012':
        for r in g5['per_seed'][s]['rows']:
            e = r['ep']
            m = json.load(open(os.path.join(M, 'g5', f's{s}', f'g5_mixed_ep{e}.json')))
            p = json.load(open(os.path.join(M, 'g5', f's{s}', f'g5_paired_ep{e}.json')))
            L.append(f"| s{s} | {e} | {ci(m['G5a']['green_K15_freeze']['g_other'])} | {ci(p['G5a_green_K15_freeze'])} | "
                     f"{ci(m['G5a']['pref_K15_freeze']['g_other'])} | {ci(p['G5a_pref_K15_freeze'])} | "
                     f"{r['values']['G5b_mixed']:+.3f} / {r['values']['G5b_paired']:+.3f} | {r['G5a']} / {r['G5b']} |")
    # 4 comparison and subsets
    sb = json.load(open(os.path.join(M, 'subsets', 'subsets_v13.json')))
    L += ['', '## 4. v12 との比較（ログ §3.4）と部分集合（§3.3 末尾）', '', '| 組 | ep | G3 v13−v12 | G5a ① v13−v12 | G5a ② v13−v12 | G5b v13−v12 |', '|---|---|---|---|---|---|']
    for s in '012':
        for e in EPS:
            c = sb['compare'][f's{s}_ep{e}']
            L.append(f"| s{s} | {e} | {ci(c['G3_K15_hold'])} | {ci(c['G5a_green_K15_freeze'])} | {ci(c['G5a_pref_K15_freeze'])} | {ci(c['G5b'])} |")
    L += ['', f"- 判定（§3.4）：{json.dumps({k: (v['points_per_pair'], v['stronger']) for k, v in sb['compare_verdict'].items()}, ensure_ascii=False)}",
          f"- 部分集合の大きさ：{sb['n_far']}", '',
          '| 組（ep200） | 量 | 部分集合 | v13 経験あり | v13 組の差 | v12 経験あり | v12 組の差 |', '|---|---|---|---|---|---|---|']
    for s in '012':
        a, b = sb['subsets'][f's{s}_ep200'], sb['v12_subsets'][f's{s}_ep200']
        for q in a:
            for sub in a[q]:
                L.append(f"| s{s} | {q} | {sub} | {ci(a[q][sub]['mixed'])} | {ci(a[q][sub]['paired'])} | "
                         f"{ci(b[q][sub]['mixed'])} | {ci(b[q][sub]['paired'])} |")
    # 5 tracking split
    L += ['', '## 5. 自己位置の追跡を 3 つの test で（運動が見える一歩、ep200。R² / 位置の誤差の中央値 / フレーム数。ログ §7・§11）', '',
          '| 組 | モデル | test | 全体 | 赤から 5 以内 | それ以外 |', '|---|---|---|---|---|---|']
    for s in '012':
        for tag, root, c in (('経験なし（v12）', os.path.join(M, 'track'), 'random'), ('経験あり 50%（v12）', os.path.join(R, 'v13', 'v12ref', 'track'), 'mixed'),
                             ('経験あり 80%（v13）', os.path.join(M, 'track'), 'mixed')):
            d = json.load(open(os.path.join(root, f's{s}', f'track_{c}_ep200.json')))
            for t in ('common', 'mixed50', 'mixed80'):
                v = d[t]['visible']
                L.append(f"| s{s} | {tag} | {t} | " + ' | '.join(f"{v[k]['r2']:.3f} / {v[k]['err_median']:.2f} / {v[k]['n']}" for k in ('all', 'near_red', 'away')) + ' |')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
