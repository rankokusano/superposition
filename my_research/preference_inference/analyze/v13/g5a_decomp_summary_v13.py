"""
Summary of the G5a decomposition (docs/v13_experiment_log.md Sec.15; post-hoc, gates unchanged).
Per pair, late-5 checkpoint, set (green = (1), pref = (2)) and model (random / mixed50 / mixed80):
  freeze    gate g_other (saved per episode by eval_g5_v13)
  a_zero    g(freeze) - g(zero from t0)             (from the same saved files)
  b_flip    g with -P_A2(t0) held                   (g5a_decomp_v13)
  b_diff    g(freeze) - g(flip)
  c_pre     pre-freeze placement on u_liked; c_pre_lmo  liked minus the 3 others
For each: the mean [CI], the paired difference vs no experience, and v13 - v12 (mixed80 - mixed50).
Counts per pair: late-5 points where the mean is > 0 with CI above 0 ('pos') or < 0 with CI below 0 ('neg').
Writes data/result/v13/post/g5a_decomp/summary.json; prints markdown tables.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES, boot_ci  # noqa

EPS = (160, 170, 180, 190, 200)
SETS = ('green', 'pref')
QS = ('freeze', 'a_zero', 'b_flip', 'b_diff', 'c_pre', 'c_pre_lmo')


def summ(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def per_episode(s, e):
    gate = np.load(os.path.join(RES, 'v13', 'main', 'g5', f's{s}', f'g5_per_ep{e}.npz'))
    ref = np.load(os.path.join(RES, 'v13', 'v12ref', 'g5', f's{s}', f'g5_per_ep{e}.npz'))
    src = {'random': (gate, 'random'), 'mixed50': (ref, 'mixed'), 'mixed80': (gate, 'mixed')}
    out = {}
    for c, (f, key) in src.items():
        dd = np.load(os.path.join(RES, 'v13', 'post', 'g5a_decomp', f's{s}', f'decomp_{c}_ep{e}.npz'))
        for k in SETS:
            fr, zr = f[f'{key}__G5a_{k}_K15_freeze'], f[f'{key}__G5a_{k}_K15_zero']
            out[(c, k)] = {'freeze': fr, 'a_zero': fr - zr, 'b_flip': dd[f'{k}_flip_g'], 'b_diff': fr - dd[f'{k}_flip_g'],
                           'c_pre': dd[f'{k}_pre_liked'], 'c_pre_lmo': dd[f'{k}_pre_liked_minus_others']}
    return out


def main():
    res, cnt = {}, {}
    for s in (0, 1, 2):
        for e in EPS:
            pe = per_episode(s, e)
            for k in SETS:
                for q in QS:
                    r = {c: summ(pe[(c, k)][q]) for c in ('random', 'mixed50', 'mixed80')}
                    r['paired50'] = summ(pe[('mixed50', k)][q] - pe[('random', k)][q])
                    r['paired80'] = summ(pe[('mixed80', k)][q] - pe[('random', k)][q])
                    r['v13_minus_v12'] = summ(pe[('mixed80', k)][q] - pe[('mixed50', k)][q])
                    res[f's{s}_ep{e}_{k}_{q}'] = r
                    for col, x in r.items():
                        c = cnt.setdefault(f'{k}_{q}_{col}', {f's{t}': {'pos': 0, 'neg': 0} for t in (0, 1, 2)})[f's{s}']
                        c['pos'] += x['mean'] > 0 and x['ci95'][0] > 0
                        c['neg'] += x['mean'] < 0 and x['ci95'][1] < 0
    ref = np.load(os.path.join(RES, 'v13', 'post', 'g5a_decomp', 'ref_true_view.npz'))
    out = {'per_ckpt': res, 'counts': cnt, 'ref_true_view': {k: summ(ref[k]) for k in ref.files}}
    with open(os.path.join(RES, 'v13', 'post', 'g5a_decomp', 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)

    def rng(k, q, col):
        v = [res[f's{s}_ep{e}_{k}_{q}'][col]['mean'] for s in (0, 1, 2) for e in EPS]
        return f'{min(v):+.2f}〜{max(v):+.2f}'

    def pts(k, q, col, sign):
        c = cnt[f'{k}_{q}_{col}']
        return '/'.join(str(c[f's{t}'][sign]) for t in (0, 1, 2))
    names = {'freeze': '止めた傾き（関門）', 'a_zero': '(a) 止める − 零', 'b_flip': '(b) 逆にした快・不快の傾き',
             'b_diff': '(b) 止める − 逆', 'c_pre': '(c) 止める前の位置（好きな色の方向）', 'c_pre_lmo': '(c) 同、好きな色 − ほか 3 色'}
    for k in SETS:
        print(f"\n**{'①' if k == 'green' else '②'}**（late-5 × 3 組の平均の範囲。［ ］は正で CI が 0 を含まない点／負で CI が 0 を含まない点の数、s0/s1/s2）\n")
        print('| 量 | 経験なし | 50% 経験あり | 50% 組の差 | 80% 経験あり | 80% 組の差 | v13 − v12 |')
        print('|---|---|---|---|---|---|---|')
        for q in QS:
            cells = []
            for col in ('random', 'mixed50', 'paired50', 'mixed80', 'paired80', 'v13_minus_v12'):
                cells.append(f"{rng(k, q, col)} ［{pts(k, q, col, 'pos')}｜{pts(k, q, col, 'neg')}］")
            print(f'| {names[q]} | ' + ' | '.join(cells) + ' |')
    print('\n- 読み取りの参照（本当の視界、t0）：' + ', '.join(f"{k} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] (n {v['n']})" for k, v in out['ref_true_view'].items()))


if __name__ == '__main__':
    main()
