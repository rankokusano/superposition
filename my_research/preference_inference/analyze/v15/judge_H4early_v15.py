"""
v15 Sec.43.4 H4 before overfitting (descriptive only; the H4 judgement stays 'does not help' at ep1500, Sec.43.1). The two
ways of choosing the checkpoint were fixed in docs/v15_experiment_log.md Sec.43.4 before any pair accuracy at these epochs:
    ep200:   epoch 200 for all 6 runs
    testmin: per run, the epoch with the lowest self-vision loss on the 300 unused test episodes (log/test/self_vision.log;
             loss only, ties -> the earliest epoch)
Same measures as judge_H4_v15.py (pair accuracy, exp - noexp, X / Y side, m, d, G 10-step readout), CPU evaluations
hide_<name>_ep<E>_cpu.json from run_cpu_H4early.sh.

    python analyze/v15/judge_H4early_v15.py --select   # print the epochs of both ways (losses only)
    python analyze/v15/judge_H4early_v15.py            # -> data/result/v15/stage2/judge_H4early_v15i_cpu.json
"""

import json
import os
import sys

import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
import analyze.v15.judge_AB_v15 as AB  # noqa
from analyze.v15.judge_H4_v15 import SEEDS, compare, load, losses  # noqa

STEMS = ('v15_s2p2h_exp_v15i_o100', 'v15_s2p2h_noexp_v15i_o100')


def run(stem, s):
    return stem + ('' if s == 0 else f'_s{s}')


def select():
    """-> {way: {(stem, seed): epoch}} from the test self-vision losses only."""
    out = {'ep200': {}, 'testmin': {}}
    for stem in STEMS:
        for s in SEEDS:
            te = losses(stem, s, 'test')
            lo = min(te.values())
            out['ep200'][(stem, s)] = 200
            out['testmin'][(stem, s)] = min(e for e, v in te.items() if v == lo)
    return out


def load_at(stem, ep):
    AB.path = lambda exp, seed: os.path.join(
        AB.S2, f"hide_{exp if seed == 0 else f'{exp}_seed{seed}'}_ep{ep[seed]}_cpu.json")
    return load(stem)


def main():
    sel = select()
    if '--select' in sys.argv:
        for way, d in sel.items():
            for (stem, s), e in d.items():
                print(way, run(stem, s), s, e)
        return
    R = {'source': 'cpu', 'descriptive_only': True,
         'epochs': {w: {f'{run(st, s)}': e for (st, s), e in d.items()} for w, d in sel.items()}}
    for way, d in sel.items():
        E = load_at(STEMS[0], {s: d[(STEMS[0], s)] for s in SEEDS})
        N = load_at(STEMS[1], {s: d[(STEMS[1], s)] for s in SEEDS})
        for s in SEEDS:
            assert np.array_equal(E[s]['ids'], N[s]['ids'])
        H = compare(E, N)
        R[way] = {'per_seed': H,
                  'mean_exp_minus_noexp': float(np.mean([H[s]['exp_minus_noexp']['mean'] for s in SEEDS])),
                  'n_seeds_ci_lower_above_0': {k: int(sum(H[s][f'exp_minus_noexp{x}']['ci95'][0] > 0 for s in SEEDS))
                                               for k, x in (('pair', ''), ('X', '_X'), ('Y', '_Y'))},
                  'n_seeds_ci_upper_below_0': int(sum(H[s]['exp_minus_noexp']['ci95'][1] < 0 for s in SEEDS))}
    out = os.path.join(AB.S2, 'judge_H4early_v15i_cpu.json')
    json.dump(R, open(out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print(json.dumps({w: {k: v for k, v in R[w].items() if k != 'per_seed'} for w in sel}, indent=1))


if __name__ == '__main__':
    main()
