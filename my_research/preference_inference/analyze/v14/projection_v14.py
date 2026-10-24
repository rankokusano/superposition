"""
v14 Sec.25 (pre-registered, descriptive): projection of A-1's own preference. For a model evaluated with A-1 liking
Red (default outputs) and Green (--a1_liked green, tag _a1green), per episode of the pref test:
  Delta = choice_c+(Green) - choice_c+(Red)   (choice 1 / 0 / 0.5 on a tie; T = 50)
  Delta' = Delta * (+1 in layout 1, -1 in layout 0)   (> 0 = the choice moved to A-1's new liked colour's side:
           Green is on bins 0,1 = c- in layout 0 and on bins 6,7 = c+ in layout 1)
mean Delta' with a 1000 episode bootstrap 95% CI, per cell (layout x A-2 liked) Delta and Delta'; for the self-vision
L1 (main) and FP. Rule (late-5): >= 3 of 5 points with CI > 0 -> 'projection', < 0 -> 'contrast', else 'no mixing'.
Writes data/result/v14/inverse/<name>.json.
"""

import argparse
import json
import os

import numpy as np

from aggregate_inverse_v14 import BASE, EPOCHS, choice, ci

SUB = os.path.join(BASE, 'inverse')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True, help='model tags, e.g. v14c_mixedpref_0 v12_mixed_0')
    ap.add_argument('--epochs', type=int, nargs='+', default=list(EPOCHS))
    ap.add_argument('--name', default='projection')
    a = ap.parse_args()
    out = {}
    for tag in a.tags:
        for q in ('L1', 'FP'):
            n_pos = n_neg = 0
            rows = {}
            for e in a.epochs:
                zr = np.load(os.path.join(SUB, f'{tag}_ep{e}_pref.npz'))
                zg = np.load(os.path.join(SUB, f'{tag}_ep{e}_a1green_pref.npz'))
                assert np.array_equal(zr['correct_sign'], zg['correct_sign'])
                dlt = choice(zg, q, 50) - choice(zr, q, 50)
                sgn = np.where(zr['layout'] == 1, 1, -1)
                dp = dlt * sgn
                rng = np.random.RandomState(0)
                c = ci([dp[rng.randint(0, len(dp), len(dp))].mean() for _ in range(1000)])
                cell = zr['layout'].astype(int) * 2 + zr['a2_liked'].astype(int)
                rows[e] = {'delta_prime': float(dp.mean()), 'ci': c,
                           'delta_by_cell': {int(k): float(dlt[cell == k].mean()) for k in range(4)},
                           'delta_prime_by_cell': {int(k): float(dp[cell == k].mean()) for k in range(4)},
                           'frac_changed': float((dlt != 0).mean())}
                n_pos += c[0] > 0
                n_neg += c[1] < 0
            word = 'projection' if n_pos >= 3 else ('contrast' if n_neg >= 3 else 'no mixing')
            out[f'{tag}|{q}'] = {'points': rows, 'n_pos': int(n_pos), 'n_neg': int(n_neg), 'word': word}
            print(f"{tag:22s} {q}: delta' {[round(r['delta_prime'], 3) for r in rows.values()]} "
                  f"CI>0 {n_pos}/{len(rows)} CI<0 {n_neg}/{len(rows)} -> {word}  changed {np.mean([r['frac_changed'] for r in rows.values()]):.2f}")
    json.dump(out, open(os.path.join(SUB, a.name + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
