"""
v14-A, note 5 follow-up (docs/v14_experiment_log.md Sec.11; NOT pre-registered, no new model runs).
From the saved v14-A outputs (data/result/v14/inverse/*_ep{e}_pref.npz), for choosing by the other-side
feature-prediction error (FP) and, for comparison, by the self-vision L1:
  * BA of the experienced and no-experience models (T = 50, 100), with episode-bootstrap CIs;
  * c0 comparison: cumulative error of c0 minus that of the correct candidate, and minus cp / cm separately;
  * choice bias: share of episodes choosing c+ per cell (layout x A-2 liked), and the same for the
    reference choice (ref_red vs ref_green -> which side it puts as pleasant).
Writes data/result/v14/inverse/fp_check.json.
"""

import json
import os

import numpy as np

from aggregate_inverse_v14 import BASE, EPOCHS, N_BOOT, ba, boot, cells, choice, ci, correct

SUB = os.path.join(BASE, 'inverse')
PAIRS = [(f'v12_mixed_{s}', f'v12_random_{s}') for s in range(4)]
CELL_NAMES = {0: 'L0_green(c-)', 1: 'L0_red(c+)', 2: 'L1_green(c+)', 3: 'L1_red(c-)'}


def ref_plus(z, q, T):
    """reference choice expressed as 'puts bins 6,7 pleasant' (= c+ side): ref_red is c+ side in layout 0."""
    ch = choice(z, q, T, 'ref_red', 'ref_green')        # 1 = ref_red chosen
    return np.where(z['layout'] == 0, ch, 1 - ch)


def one(z, q, T, B):
    cl = cells(z)
    c = correct(z, q, T)
    r = {'ba': ba(c, cl), 'ba_ci': ci([ba(c, cl, i) for i in B])}
    cum = {k: z[f'{q}_{k}'][:, :T].sum(1) for k in ('cp', 'cm', 'c0')}
    ecor = np.where(z['correct_sign'] > 0, cum['cp'], cum['cm'])
    for nm, v in (('c0_minus_correct', cum['c0'] - ecor), ('c0_minus_cp', cum['c0'] - cum['cp']),
                  ('c0_minus_cm', cum['c0'] - cum['cm'])):
        r[nm] = {'mean': float(v.mean()), 'ci': ci([v[i].mean() for i in B]), 'frac_pos': float((v > 0).mean())}
    chp = choice(z, q, T)
    r['frac_choose_cplus_by_cell'] = {CELL_NAMES[k]: float(chp[cl == k].mean()) for k in range(4)}
    r['frac_choose_cplus_all'] = float(chp.mean())
    rp = ref_plus(z, q, T)
    r['ref_frac_cplus_side_by_cell'] = {CELL_NAMES[k]: float(rp[cl == k].mean()) for k in range(4)}
    return r


def main():
    out = {}
    for ex, nx in PAIRS:
        for e in EPOCHS:
            ze = np.load(os.path.join(SUB, f'{ex}_ep{e}_pref.npz'))
            zr = np.load(os.path.join(SUB, f'{nx}_ep{e}_pref.npz'))
            B = boot(cells(ze), len(ze['correct_sign']))
            for q in ('FP', 'L1'):
                for T in (50, 100):
                    out[f'{ex}|{q}|T{T}|ep{e}'] = {'exp': one(ze, q, T, B), 'noexp': one(zr, q, T, B)}
    with open(os.path.join(SUB, 'fp_check.json'), 'w') as f:
        json.dump(out, f, indent=1)
    for ex, _ in PAIRS:
        for q in ('FP', 'L1'):
            r = out[f'{ex}|{q}|T50|ep200']
            print(ex, q, 'T50 ep200')
            for side in ('exp', 'noexp'):
                v = r[side]
                print(f"  {side:5s} BA {v['ba']:.3f} {np.round(v['ba_ci'], 3).tolist()}  c0-correct {v['c0_minus_correct']['mean']:+.3f} "
                      f"{np.round(v['c0_minus_correct']['ci'], 3).tolist()} (pos {v['c0_minus_correct']['frac_pos']:.2f})  "
                      f"c0-cp {v['c0_minus_cp']['mean']:+.3f} c0-cm {v['c0_minus_cm']['mean']:+.3f}  "
                      f"P(c+) {v['frac_choose_cplus_all']:.2f} by cell {[round(x, 2) for x in v['frac_choose_cplus_by_cell'].values()]}  "
                      f"ref c+side {[round(x, 2) for x in v['ref_frac_cplus_side_by_cell'].values()]}")


if __name__ == '__main__':
    main()
