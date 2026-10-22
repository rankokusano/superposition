"""
v14-A FP follow-up 2 (docs/v14_experiment_log.md Sec.18; NOT pre-registered, no new model runs): does the choice
by the other-side feature-prediction error (FP; self-vision L1 for comparison) follow A-2's preference or A-2's
recent direction of travel? From the saved v14-A outputs (v12 pairs s0..s3, pref test) and the pref test data:
  direction answer: c+ if A-2's displacement over t = 1..50 (pos[51] - pos[1]) projects further on the bins-6,7
                    direction (+135 deg) than on the bins-0,1 direction (-135 deg), else c-;
  preference answer: correct_sign (v14-A Sec.3 C).
Per model / checkpoint / subset (all episodes; the lowest third of A-2's rule-step share among awake steps in
t = 1..50): agreement of the choice with each answer (raw, 1000 episode bootstrap 95% CI), agreement of the two
answers themselves, and among episodes where the answers disagree, the share of choices that follow preference.
Writes data/result/v14/inverse/fp_direction.json.
"""

import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
from aggregate_inverse_v14 import BASE, EPOCHS, choice, ci  # noqa
from eval_s2_v11 import load_set  # noqa
from v11core.pleasure import BIN_UNIT  # noqa

SUB = os.path.join(BASE, 'inverse')
T = 50


def stats(chp, pref, dirn, idx, rng):
    """chp: 1 if c+ chosen (0.5 tie); pref, dirn: +1 / -1 answers; idx: episode subset."""
    c, p, d = chp[idx], pref[idx], dirn[idx]
    agp = np.where(p > 0, c, 1 - c)
    agd = np.where(d > 0, c, 1 - c)
    B = [rng.randint(0, len(idx), len(idx)) for _ in range(1000)]
    dis = p != d
    r = {'n': int(len(idx)), 'agree_pref': float(agp.mean()), 'agree_pref_ci': ci([agp[b].mean() for b in B]),
         'agree_dir': float(agd.mean()), 'agree_dir_ci': ci([agd[b].mean() for b in B]),
         'answers_agree': float((p == d).mean()), 'n_disagree': int(dis.sum())}
    if dis.sum():
        fp = agp[dis]
        Bd = [rng.randint(0, dis.sum(), dis.sum()) for _ in range(1000)]
        r['disagree_follow_pref'] = float(fp.mean())
        r['disagree_follow_pref_ci'] = ci([fp[b].mean() for b in Bd])
    return r


def main():
    d = load_set('pref')
    g = h5py.File(d['path'], 'r')[d['split']]
    disp = d['other_position'][:, 51] - d['other_position'][:, 1]
    u01, u67 = BIN_UNIT[[0, 1]].mean(0), BIN_UNIT[[6, 7]].mean(0)
    u01, u67 = u01 / np.linalg.norm(u01), u67 / np.linalg.norm(u67)
    dirn = np.where(disp @ u67 > disp @ u01, 1, -1)
    src = g['a2_source'][:, 1:51]
    aw = src != 3
    rule = ((src == 1) & aw).sum(1) / np.maximum(aw.sum(1), 1)
    low = np.where(rule <= np.percentile(rule, 100 / 3))[0]
    allep = np.arange(len(rule))
    out = {'definition': 'Sec.18', 'rule_share_low_third_max': float(np.percentile(rule, 100 / 3)),
           'n_low': int(len(low)), 'u01': u01.tolist(), 'u67': u67.tolist()}
    for s in range(4):
        for cond in ('mixed', 'random'):
            for e in EPOCHS:
                z = np.load(os.path.join(SUB, f'v12_{cond}_{s}_ep{e}_pref.npz'))
                pref = z['correct_sign']
                for q in ('FP', 'L1'):
                    chp = choice(z, q, T)
                    for nm, idx in (('all', allep), ('low_rule', low)):
                        out[f'{cond}_s{s}|{q}|{nm}|ep{e}'] = stats(chp, pref, dirn, idx, np.random.RandomState(0))
    json.dump(out, open(os.path.join(SUB, 'fp_direction.json'), 'w'), indent=1)
    print('low-third rule share <=', round(out['rule_share_low_third_max'], 3), 'n', out['n_low'])
    for q in ('FP', 'L1'):
        for nm in ('all', 'low_rule'):
            print(f'== {q} {nm}')
            for s in range(4):
                for cond in ('mixed', 'random'):
                    r = out[f'{cond}_s{s}|{q}|{nm}|ep200']
                    late = [out[f'{cond}_s{s}|{q}|{nm}|ep{e}'] for e in EPOCHS]
                    rp = (min(x['agree_pref'] for x in late), max(x['agree_pref'] for x in late))
                    rd = (min(x['agree_dir'] for x in late), max(x['agree_dir'] for x in late))
                    rf = (min(x.get('disagree_follow_pref', np.nan) for x in late), max(x.get('disagree_follow_pref', np.nan) for x in late))
                    print(f"  s{s} {cond:6s} pref {r['agree_pref']:.3f} {np.round(r['agree_pref_ci'], 3).tolist()} [{rp[0]:.3f}-{rp[1]:.3f}]  "
                          f"dir {r['agree_dir']:.3f} {np.round(r['agree_dir_ci'], 3).tolist()} [{rd[0]:.3f}-{rd[1]:.3f}]  "
                          f"answers agree {r['answers_agree']:.3f}  disagree n={r['n_disagree']} follow pref "
                          f"{r.get('disagree_follow_pref', float('nan')):.3f} {np.round(r.get('disagree_follow_pref_ci', [np.nan, np.nan]), 3).tolist()} [{rf[0]:.3f}-{rf[1]:.3f}]")


if __name__ == '__main__':
    main()
