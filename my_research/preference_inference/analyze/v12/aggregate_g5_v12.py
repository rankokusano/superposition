"""
v12 gate G5 judgement (docs/v12_experiment_log.md Sec.13) from eval_g5_v12.py outputs
data/result/v12/<root>/s<seed>/g5_{random,mixed}_ep<e>.json and g5_paired_ep<e>.json.
Per checkpoint: G5a passes when, in BOTH sets (1: green-liker, 2: preference per episode; K = 15,
freeze), the experienced model's g_other CI lower > 0 and the paired (mixed - random) CI lower > 0;
G5b passes when the experienced model's main value CI lower > 0 and the paired CI lower > 0.
A pair passes with >= --min_ckpt checkpoints, the gate with >= --min_seed pairs.
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v12 import RES  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='main/g5')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--flat', action='store_true')
    ap.add_argument('--min_ckpt', type=int, default=3)
    ap.add_argument('--min_seed', type=int, default=2)
    args = ap.parse_args()
    root = os.path.join(RES, 'v12', args.root)
    J = lambda s, f: json.load(open(os.path.join(root if args.flat else os.path.join(root, f's{s}'), f)))  # noqa
    res = {'per_seed': {}}
    for s in args.seeds:
        rows = []
        for e in args.epochs:
            m, p = J(s, f'g5_mixed_ep{e}.json'), J(s, f'g5_paired_ep{e}.json')
            a = {k: bool(m['G5a'][f'{k}_K15_freeze']['g_other']['ci95'][0] > 0 and p[f'G5a_{k}_K15_freeze']['ci95'][0] > 0)
                 for k in ('green', 'pref')}
            # secondary (Sec.22, B1 background-relative read-out); not used for the stage-2 decision
            ab = {k: bool(m['G5a_bg'][f'{k}_K15_freeze']['g_other']['ci95'][0] > 0 and p[f'G5a_bg_{k}_K15_freeze']['ci95'][0] > 0)
                  for k in ('green', 'pref')} if 'G5a_bg' in m else {'green': False, 'pref': False}
            bb = bool(m['G5b']['bg']['main_pm_own']['ci95'][0] > 0 and p['G5b_bg']['ci95'][0] > 0) if 'bg' in m['G5b'] else False
            rows.append({'ep': e, 'G5a': a['green'] and a['pref'], 'G5a_per_set': a,
                         'G5a_bg': ab['green'] and ab['pref'], 'G5b_bg': bb,
                         'G5b': bool(m['G5b']['main_pm_own']['ci95'][0] > 0 and p['G5b']['ci95'][0] > 0),
                         'values': {'G5a_mixed': {k: m['G5a'][f'{k}_K15_freeze']['g_other']['mean'] for k in ('green', 'pref')},
                                    'G5a_paired': {k: p[f'G5a_{k}_K15_freeze']['mean'] for k in ('green', 'pref')},
                                    'G5b_mixed': m['G5b']['main_pm_own']['mean'], 'G5b_paired': p['G5b']['mean']}})
        res['per_seed'][str(s)] = {'rows': rows, 'seed_pass': {k: sum(r[k] for r in rows) >= args.min_ckpt for k in ('G5a', 'G5b', 'G5a_bg', 'G5b_bg')}}
    res['gate'] = {k: sum(res['per_seed'][str(s)]['seed_pass'][k] for s in args.seeds) >= args.min_seed for k in ('G5a', 'G5b')}
    res['secondary_bg'] = {k: sum(res['per_seed'][str(s)]['seed_pass'][k] for s in args.seeds) >= args.min_seed for k in ('G5a_bg', 'G5b_bg')}
    json.dump(res, open(os.path.join(root, 'gate_g5.json'), 'w'), indent=1)
    print(json.dumps(res['gate']), json.dumps(res['secondary_bg']), json.dumps({s: v['seed_pass'] for s, v in res['per_seed'].items()}))


if __name__ == '__main__':
    main()
