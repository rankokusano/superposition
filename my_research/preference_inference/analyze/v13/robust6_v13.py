"""
v13 addition: robustness over all 6 pairs s0-s5 (docs/v13_experiment_log.md Sec.31, pre-registered).
Per pair and late-5 checkpoint, the gate rule (5 points, >= 3 with CIs excluding 0):
  (a) self side, experienced 50% and 80%: G2 = M1(c) (mean < 0, paired CI < 0), M1(d) (mean > 0, paired CI > 0),
      colour (CI > 0), all three; G3 (experienced CI > 0 and paired CI > 0)  [rules of aggregate_s1_v12.py]
  (b) within-seed dose 80% - 50% (same episodes) > 0 with CI excluding 0: G3, G5b, sign-following part (1), (2)
  (c) difference to no experience: experienced CI > 0 and paired (experienced - none) CI > 0: G5b, sign part (1), (2);
      for 80% and for 50%
  (d) order of the G5b late-5 means: none < 50% < 80% (reported as is)
Descriptive: G1 visible-step R^2, G5a gate (both sets), h2 -> A-2 position, D6-like secondary (ep200).
Sources: 80% pair = data/result/v13/main/{s1eval,g5,h2_check_s*,d6}; 50%: s0-s3 = data/result/v12/main/s1eval and
data/result/v13/v12ref/g5, s4-s5 = data/result/v13/v12ref/{s1eval,g5,h2_check_s*,d6}; sign part = post/g5a_decomp.
Writes data/result/v13/extra/robust6.json; prints markdown.
"""

import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v11 import RES, boot_ci  # noqa

EPS = (160, 170, 180, 190, 200)
SEEDS = (0, 1, 2, 3, 4, 5)
J = lambda *a: json.load(open(os.path.join(RES, *a)))  # noqa
N = lambda *a: np.load(os.path.join(RES, *a))  # noqa


def s1root(cond, s):
    if cond == '80':
        return ('v13', 'main', 's1eval', f's{s}')
    return ('v12', 'main', 's1eval', f's{s}') if s <= 3 else ('v13', 'v12ref', 's1eval', f's{s}')


def summ(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def pos(x):
    return x['mean'] > 0 and x['ci95'][0] > 0


def main():
    res = {}
    for s in SEEDS:
        r = {k: 0 for k in ('a_G2_50', 'a_G2_80', 'a_G3_50', 'a_G3_80', 'b_G3', 'b_G5b', 'b_sign_green', 'b_sign_pref',
                            'c_G5b_80', 'c_G5b_50', 'c_sign_green_80', 'c_sign_green_50', 'c_sign_pref_80', 'c_sign_pref_50',
                            'G1_80', 'G1_50', 'G5a_80', 'G5a_50')}
        means = {'G5b': {'none': [], '50': [], '80': []}}
        for e in EPS:
            for cond in ('50', '80'):
                root = s1root(cond, s)
                m, p = J(*root, f'eval_mixed_ep{e}.json'), J(*root, f'paired_ep{e}.json')
                g2 = (m['M1']['hidden']['M1c_c_pred']['mean'] < 0 and p['hidden_c_pred']['ci95'][1] < 0 and
                      m['M1']['hidden']['M1d_c_cf']['mean'] > 0 and p['hidden_c_cf']['ci95'][0] > 0 and
                      m['M1']['hidden']['colour_swap_SRed_minus_SGreen']['ci95'][0] > 0)
                g3 = m['freeze']['K15_hold']['g']['ci95'][0] > 0 and p['freeze_K15_hold']['ci95'][0] > 0
                r[f'a_G2_{cond}'] += int(g2)
                r[f'a_G3_{cond}'] += int(g3)
                rnd = J(*root, f'eval_random_ep{e}.json')
                r[f'G1_{cond}'] += int(min(m['tracking']['p_next']['visible'], rnd['tracking']['p_next']['visible']) >= 0.80)
            # per-episode values
            p80, p50 = N(*s1root('80', s), f'per_ep{e}.npz'), N(*s1root('50', s), f'per_ep{e}.npz')
            r['b_G3'] += int(pos(summ(p80['mixed__freeze_K15_hold'] - p50['mixed__freeze_K15_hold'])))
            g80 = N('v13', 'main', 'g5', f's{s}', f'g5_per_ep{e}.npz')
            g50 = N('v13', 'v12ref', 'g5', f's{s}', f'g5_per_ep{e}.npz')
            x80, x50, xn = g80['mixed__G5b'], g50['mixed__G5b'], g80['random__G5b']
            r['b_G5b'] += int(pos(summ(x80 - x50)))
            r['c_G5b_80'] += int(pos(summ(x80)) and pos(summ(x80 - xn)))
            r['c_G5b_50'] += int(pos(summ(x50)) and pos(summ(x50 - xn)))
            means['G5b']['none'].append(np.nanmean(xn)); means['G5b']['50'].append(np.nanmean(x50)); means['G5b']['80'].append(np.nanmean(x80))
            for cond, gg, key in (('80', g80, 'mixed'), ('50', g50, 'mixed')):
                a = all(pos(summ(gg[f'{key}__G5a_{k}_K15_freeze'])) and pos(summ(gg[f'{key}__G5a_{k}_K15_freeze'] - g80[f'random__G5a_{k}_K15_freeze']))
                        for k in ('green', 'pref'))
                r[f'G5a_{cond}'] += int(a)
            dd = {c: N('v13', 'post', 'g5a_decomp', f's{s}', f'decomp_{c}_ep{e}.npz') for c in ('random', 'mixed50', 'mixed80')}
            for k in ('green', 'pref'):
                sp = {'none': (g80[f'random__G5a_{k}_K15_freeze'] - dd['random'][f'{k}_flip_g']) / 2,
                      '50': (g50[f'mixed__G5a_{k}_K15_freeze'] - dd['mixed50'][f'{k}_flip_g']) / 2,
                      '80': (g80[f'mixed__G5a_{k}_K15_freeze'] - dd['mixed80'][f'{k}_flip_g']) / 2}
                r[f'b_sign_{k}'] += int(pos(summ(sp['80'] - sp['50'])))
                for cond in ('80', '50'):
                    r[f'c_sign_{k}_{cond}'] += int(pos(summ(sp[cond])) and pos(summ(sp[cond] - sp['none'])))
        mm = {k: float(np.mean(v)) for k, v in means['G5b'].items()}
        r['d_G5b_means'] = mm
        r['d_order'] = bool(mm['none'] < mm['50'] < mm['80'])
        # descriptive
        h = {}
        for cond, root in (('80', ('v13', 'main')), ('50', ('v13', 'v12ref') if s >= 4 else ('v12', 'main'))):
            fs = sorted(glob.glob(os.path.join(RES, *root, f'h2_check_s{s}', '*.json')))
            if fs:
                d = json.load(open(fs[-1]))
                h[cond] = {k: [round(x, 3) for x in v['ridge_xy']] for k, v in d.items() if isinstance(v, dict) and 'ridge_xy' in v}
        r['h2'] = h
        res[f's{s}'] = r
    os.makedirs(os.path.join(RES, 'v13', 'extra'), exist_ok=True)
    json.dump(res, open(os.path.join(RES, 'v13', 'extra', 'robust6.json'), 'w'), indent=1)

    def row(name, key):
        pts = [res[f's{s}'][key] for s in SEEDS]
        return f"| {name} | " + ' | '.join(f"{p}/5{' ✓' if p >= 3 else ''}" for p in pts) + f" | **{sum(p >= 3 for p in pts)}/6**（本番 3 組 {sum(p >= 3 for p in pts[:3])}/3） |"
    print('| 項目 | s0 | s1 | s2 | s3 | s4 | s5 | 成り立つ組 |')
    print('|---|---|---|---|---|---|---|---|')
    for name, key in (('(a) G2 経験あり 50%', 'a_G2_50'), ('(a) G2 経験あり 80%', 'a_G2_80'), ('(a) G3 経験あり 50%', 'a_G3_50'),
                      ('(a) G3 経験あり 80%', 'a_G3_80'), ('(b) G3：80% − 50%', 'b_G3'), ('(b) G5b：80% − 50%', 'b_G5b'),
                      ('(b) 符号に従う分 ①：80% − 50%', 'b_sign_green'), ('(b) 符号に従う分 ②：80% − 50%', 'b_sign_pref'),
                      ('(c) G5b：80% − 経験なし', 'c_G5b_80'), ('(c) G5b：50% − 経験なし', 'c_G5b_50'),
                      ('(c) 符号に従う分 ①：80% − 経験なし', 'c_sign_green_80'), ('(c) 符号に従う分 ①：50% − 経験なし', 'c_sign_green_50'),
                      ('(c) 符号に従う分 ②：80% − 経験なし', 'c_sign_pref_80'), ('(c) 符号に従う分 ②：50% − 経験なし', 'c_sign_pref_50'),
                      ('（記述）G1 80%', 'G1_80'), ('（記述）G1 50%', 'G1_50'), ('（記述）G5a 80%', 'G5a_80'), ('（記述）G5a 50%', 'G5a_50')):
        print(row(name, key))
    print('\n**(d) G5b の順序（late-5 の平均：経験なし / 50% / 80%）**\n')
    print('| 組 | 経験なし | 50% | 80% | 経験なし < 50% < 80% |')
    print('|---|---|---|---|---|')
    for s in SEEDS:
        mm = res[f's{s}']['d_G5b_means']
        print(f"| s{s} | {mm['none']:+.3f} | {mm['50']:+.3f} | {mm['80']:+.3f} | {'はい' if res[f's{s}']['d_order'] else 'いいえ'} |")
    print(f"\n- 順に並ぶ組：{sum(res[f's{s}']['d_order'] for s in SEEDS)}/6")
    print('- h²→A-2 の位置 R²（x, y）：' + '；'.join(f"s{s} {res[f's{s}']['h2']}" for s in SEEDS))


if __name__ == '__main__':
    main()
