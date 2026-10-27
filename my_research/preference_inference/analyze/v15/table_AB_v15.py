"""
v15 (A) / (B): one table for both encoders (Kusano-san's request, docs/v15_experiment_log.md Sec.31.5), from the judgements
data/result/v15/stage2/judge_AB_{v4,v15i}_<source>.json (judge_AB_v15.py). Descriptive only; the verdicts come from judge_AB.

    python analyze/v15/table_AB_v15.py [--source cpu]   # -> data/result/v15/stage2/table_AB_<source>.md
"""

import argparse
import json
import os

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S2 = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'stage2')


def ci(d):
    return f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='cpu', choices=['cpu', 'gpu'])
    a = ap.parse_args()
    J = {e: json.load(open(os.path.join(S2, f'judge_AB_{e}_{a.source}.json'))) for e in ('v4', 'v15i')}
    L = [f'# (A)・(B)：v4 と v15 (i)（{a.source} の値、隠した検査、生成する組 930 組）', '',
         '| 量 | v4 | v15 (i) |', '|---|---|---|']
    for s in ('0', '1', '2'):
        L.append(f'| seed {s}：経験あり − 経験なし（組の正解率） | ' +
                 ' | '.join(ci(J[e]['A'][s]['exp_minus_noexp_pairacc']) for e in J) + ' |')
    for s in ('0', '1', '2'):
        L.append(f'| seed {s}：組の正解率 あり / なし | ' +
                 ' | '.join(f"{J[e]['A'][s]['exp_pairacc']['mean']:.3f} / {J[e]['A'][s]['noexp_pairacc']['mean']:.3f}" for e in J) + ' |')
    for s in ('0', '1', '2'):
        L.append(f'| seed {s}：G から好み（10 歩後）あり / なし（下限） | ' +
                 ' | '.join(f"{J[e]['A'][s]['exp_g10']['mean']:.3f} / {J[e]['A'][s]['noexp_g10']['mean']:.3f}"
                            f" ({J[e]['A'][s]['exp_g10']['ci95'][0]:.3f} / {J[e]['A'][s]['noexp_g10']['ci95'][0]:.3f})" for e in J) + ' |')
    for k, lab in (('m', 'm（青の割合の X・Y の平均）'), ('d', 'd（青の割合の X − Y）')):
        for s in ('0', '1', '2'):
            L.append(f'| seed {s}：{lab} あり / なし | ' +
                     ' | '.join(f"{J[e]['A'][s]['exp_' + k]:.3f} / {J[e]['A'][s]['noexp_' + k]:.3f}" for e in J) + ' |')
    L.append('| 符号が 3 シードで同じ | ' + ' | '.join(str(J[e]['A']['same_sign_3_seeds']) for e in J) + ' |')
    L.append('| 95% 区間が 0 を含まないシードの数 | ' + ' | '.join(str(J[e]['A']['n_seeds_ci_excludes_0']) for e in J) + ' |')
    L.append('| 好みを保つ形が 6 本すべて（G 10 歩後の下限 > 0.5） | ' + ' | '.join(str(J[e]['A']['memory_kept_all']) for e in J) + ' |')
    for k in ('m', 'd'):
        L.append(f'| 経験ありの 3 シードの {k} の幅 | ' +
                 ' | '.join('[{:.3f}, {:.3f}]'.format(*J[e]['A'][f'exp_{k}_range']) for e in J) + ' |')
        L.append(f'| 経験なしの 3 シードの {k} の幅 | ' +
                 ' | '.join('[{:.3f}, {:.3f}]'.format(*J[e]['A'][f'noexp_{k}_range']) for e in J) + ' |')
    B = {e: J[e].get('B') for e in J}
    rows = (('(B) m / d', lambda b: f"{b['m']:.3f} / {b['d']:.3f}"),
            ('(B) acc_X / acc_Y', lambda b: f"{b['acc_X']:.3f} / {b['acc_Y']:.3f}"),
            ('(B) 組の正解率', lambda b: ci(b['pairacc'])),
            ('(B) G から好み（10 歩後）', lambda b: ci(b['g10'])),
            ('(B) Δm（順位 Y − 順位 X、seed 0）', lambda b: ci(b['dm'])),
            ('(B) Δd', lambda b: ci(b['dd'])),
            ('(B) m の位置（経験ありの 3 シードの幅）', lambda b: 'below' if b['m_below_3seed_range'] else ('inside' if b['m_inside_3seed_range'] else 'above')),
            ('(B) d が経験ありの 3 シードの幅の内', lambda b: str(b['d_inside_3seed_range'])),
            ('(B) 判定', lambda b: b['verdict']))
    for lab, f in rows:
        L.append(f'| {lab} | ' + ' | '.join(f(B[e]) if B[e] else '—' for e in J) + ' |')
    out = os.path.join(S2, f'table_AB_{a.source}.md')
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
