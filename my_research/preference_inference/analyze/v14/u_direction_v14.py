"""
v14-B: does u follow A-2's preference or A-2's recent direction? (docs/v14_experiment_log.md Sec.29 for the v12
models: method fixed after seeing the v14-B judgement; Sec.30 condition (b) for the v13 models, pre-registered.)
From the saved u (data/result/v14/stage2u/u/u_<cond>_s<seed>_ep<e>.npz, pref test), no model runs:
  * episode choice: c+ if sum_{t=1..50} u_t > 0, c- if < 0 (0.5 if 0);
  * direction / preference answers and the low-rule third as in fp_direction_v14 (Sec.18);
  * agreement with each answer, and among disagreeing episodes the share following preference (bootstrap CI);
  * per-step version: for t = 20..99, sign(u_t) vs the direction of A-2's displacement over t-10..t and vs preference.
Writes data/result/v14/stage2u/<name>.json.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _HERE)
from aggregate_inverse_v14 import BASE, EPOCHS  # noqa
from eval_s2_v11 import load_set  # noqa
from fp_direction_v14 import stats  # noqa
from v11core.pleasure import BIN_UNIT  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conds', nargs='+', default=['mixed', 'random'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, nargs='+', default=list(EPOCHS))
    ap.add_argument('--sub', default='stage2u/u')
    ap.add_argument('--name', default='u_direction')
    a = ap.parse_args()
    d = load_set('pref')
    g = h5py.File(d['path'], 'r')[d['split']]
    pos = d['other_position']
    u01, u67 = BIN_UNIT[[0, 1]].mean(0), BIN_UNIT[[6, 7]].mean(0)
    u01, u67 = u01 / np.linalg.norm(u01), u67 / np.linalg.norm(u67)
    disp = pos[:, 51] - pos[:, 1]
    dirn = np.where(disp @ u67 > disp @ u01, 1, -1)
    src = g['a2_source'][:, 1:51]
    aw = src != 3
    rule = ((src == 1) & aw).sum(1) / np.maximum(aw.sum(1), 1)
    low = np.where(rule <= np.percentile(rule, 100 / 3))[0]
    allep = np.arange(len(rule))
    lay, lk = d['layout'].astype(int), d['a2_liked'].astype(int)
    pref = np.where(((lay == 0) & (lk == 1)) | ((lay == 1) & (lk == 0)), 1, -1)
    ts = np.arange(20, 100)
    dstep = pos[:, ts] - pos[:, ts - 10]                                  # (N, 80, 2)
    dir_t = np.where(dstep @ u67 > dstep @ u01, 1, -1)
    moved = np.linalg.norm(dstep, axis=-1) > 0
    out = {'definition': 'Sec.29 / Sec.30', 'n_low': int(len(low))}
    for c in a.conds:
        for s in a.seeds:
            for e in a.epochs:
                u = np.load(os.path.join(BASE, a.sub, f'u_{c}_s{s}_ep{e}.npz'))['pref_u']
                su = u[:, 1:51].sum(1)
                chp = np.where(su > 0, 1.0, np.where(su < 0, 0.0, 0.5))
                key = f'{c}_s{s}|ep{e}'
                for nm, idx in (('all', allep), ('low_rule', low)):
                    out[f'{key}|{nm}'] = stats(chp, pref, dirn, idx, np.random.RandomState(0))
                sg = np.sign(u[:, ts])
                out[f'{key}|step'] = {'agree_dir': float((sg == dir_t)[moved].mean()),
                                      'agree_pref': float((sg == pref[:, None])[moved].mean()),
                                      'answers_agree': float((dir_t == pref[:, None])[moved].mean())}
    json.dump(out, open(os.path.join(BASE, a.sub, '..', a.name + '.json'), 'w'), indent=1)
    for c in a.conds:
        for s in a.seeds:
            for nm in ('all', 'low_rule'):
                rows = [out[f'{c}_s{s}|ep{e}|{nm}'] for e in a.epochs]
                r = rows[-1]
                print(f"{c}_s{s} {nm:8s} ep{a.epochs[-1]} pref {r['agree_pref']:.3f} dir {r['agree_dir']:.3f} "
                      f"follow pref {r['disagree_follow_pref']:.3f} {np.round(r['disagree_follow_pref_ci'], 3).tolist()} "
                      f"over epochs [{min(x['disagree_follow_pref'] for x in rows):.3f}-{max(x['disagree_follow_pref'] for x in rows):.3f}] "
                      f"CI low > 0.5: {sum(x['disagree_follow_pref_ci'][0] > 0.5 for x in rows)}/{len(rows)}")
            st = out[f'{c}_s{s}|ep{a.epochs[-1]}|step']
            print(f"   step: agree dir {st['agree_dir']:.3f} pref {st['agree_pref']:.3f} (answers agree {st['answers_agree']:.3f})")


if __name__ == '__main__':
    main()
