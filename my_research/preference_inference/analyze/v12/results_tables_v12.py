"""
v12 numbers log, stage 2 (docs/v12_results_log.md Sec.5): Markdown tables generated from the saved aggregations and
evaluator outputs (no new computation): primary T2 / T5 per attempt and pair (late-5 ranges, checkpoints in the
passing form), corr(Q2, P_A2), T3 per attempt, sign agreement (2b, shape A, 2c). Prints to stdout.
"""

import json
import os

import numpy as np

_PI = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V = os.path.join(_PI, 'data', 'result', 'v12')
EPS = (160, 170, 180, 190, 200)
ATT = [('正式', 'stage2', 'aggregate_primary_s0s1s2s3.json', (0, 1, 2, 3)),
       ('2 段目 b', 'stage2b', 'aggregate_primary_s0s1s2s3.json', (0, 1, 2, 3)),
       ('案 A（参考）', 'stage2a', 'aggregate_primary_interim_s0.json', (0,)),
       ('2 段目 c（隠した test、主）', 'stage2c', 'aggregate_primary_s0s1s2.json', (0, 1, 2)),
       ('2 段目 c（参考：隠した test の見えている区間）', 'stage2c_hidvis', 'aggregate_primary_s0s1s2.json', (0, 1, 2)),
       ('2 段目 c（参考：見えている test）', 'stage2c_vis', 'aggregate_primary_s0s1s2.json', (0, 1, 2))]


def rng(v):
    return f'{min(v):+.3f}〜{max(v):+.3f}'


def primary():
    print('| 試み | 組 | ① T2 合格の形 | ① 経験あり | ① 経験なし | ① 組の差 | ② T5 合格の形 | ② 経験あり | ② 経験なし | ② 組の差 |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for name, out, f, seeds in ATT:
        a = json.load(open(os.path.join(V, out, 'eval', f)))
        for s in seeds:
            cells = []
            for key in ('green:sT2_t50', 'pref:dT5'):
                r = a[key][f's{s}']; pc = r['per_ckpt']
                cells += [f'{r["n_pass"]}/5', rng([x['mixed'] for x in pc]), rng([x['random'] for x in pc]), rng([x['diff'] for x in pc])]
            print(f'| {name} | s{s} | ' + ' | '.join(cells) + ' |')
        jud = {k: a[k].get('setting_pass_v12_rule') for k in ('green:sT2_t50', 'pref:dT5')}
        print(f'| {name} | 判定 | {"合格" if jud["green:sT2_t50"] else "不合格"} | | | | {"合格" if jud["pref:dT5"] else "不合格"} | | | |')


def corr():
    print('| 試み | 組 | ① のセット 経験なし / 経験あり | ② のセット 経験なし / 経験あり |')
    print('|---|---|---|---|')
    for name, out, f, seeds in ATT[:4]:
        for s in seeds:
            cells = []
            for st, key in (('green', 'green_set'), ('pref', 'pref_test')):
                v = []
                for c in ('random', 'mixed'):
                    v.append(rng([json.load(open(os.path.join(V, out, 'eval', f'{st}_{c}_s{s}_ep{e}.json')))['corr_Q2_PA2'][key]['all'] for e in EPS]))
                cells.append(' / '.join(v))
            print(f'| {name} | s{s} | ' + ' | '.join(cells) + ' |')


def t3():
    print('| 試み | 指標 | 零 | 生成した Q2 | D6（P_A2） | 位置の代理 |')
    print('|---|---|---|---|---|---|')
    for name, out, seeds in (('正式', 'stage2', (0, 1, 2, 3)), ('2 段目 b', 'stage2b', (0, 1, 2, 3)), ('案 A', 'stage2a', (0,)),
                             ('2 段目 c（見えている test）', 'stage2c_vis', (0, 1, 2))):
        for metric, ml in (('other_fp_mse', '他者側の特徴予測（MSE）'), ('self_L1', '自己の視覚（L1）')):
            vals = {k: [] for k in ('zero', 'gen', 'd6', 'proxy')}
            for s in seeds:
                for st in ('green', 'pref'):
                    for c in ('random', 'mixed'):
                        js = [json.load(open(os.path.join(V, out, 't1t3', f'{st}_{c}_s{s}_ep{e}.json')))[st] for e in EPS]
                        for k in vals:
                            vals[k].append(np.mean([j[k][metric] for j in js]))
            print(f'| {name} | {ml} | ' + ' | '.join(f'{min(v):.3f}〜{max(v):.3f}' for v in vals.values()) + ' |')


def sign():
    print('| 試み | 組 | モデル | 全体 | 元の配置 | 入れ替え | u が正の割合 |')
    print('|---|---|---|---|---|---|---|')
    for name, f in (('2 段目 b', 'stage2b/summary/s2b_extra_02.json'), ('2 段目 b', 'stage2b/summary/s2b_extra_1.json'),
                    ('2 段目 b', 'stage2b/summary/s2b_extra_3.json'), ('案 A', 'stage2a/summary/s2b_extra_0.json')):
        p = os.path.join(V, f)
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        for k, v in j.items():
            if not isinstance(v, dict) or 'sign_agree_all' not in v:
                continue
            print(f'| {name} | {k.split("_")[-1]} | {"_".join(k.split("_")[:2])} | {v["sign_agree_all"]:.2f} | {v["sign_agree_original"]:.2f} | '
                  f'{v["sign_agree_swap_rg"]:.2f} | {v.get("frac_u_positive", float("nan")):.2f} |')
    for s in (0, 1, 2):
        for st in ('green', 'pref'):
            for c in ('random', 'mixed'):
                sa = json.load(open(os.path.join(V, 'stage2c', 'eval', f'{st}_{c}_s{s}_ep200.json')))['sign_agreement']
                for fr in ('hidden', 'visible'):
                    x = sa[f'green_{fr}']
                    print(f'| 2 段目 c（{"隠れている" if fr == "hidden" else "見えている"}フレーム、①のセット、ep200） | s{s} | {st}_{c} | {x["all"]:.2f} | {x["original"]:.2f} | {x["swap_rg"]:.2f} | — |')


if __name__ == '__main__':
    for nm, fn in (('primary', primary), ('corr', corr), ('t3', t3), ('sign', sign)):
        print(f'\n<!-- {nm} -->'); fn()
