"""
v14-B main measures (docs/v14_experiment_log.md Sec.4, 10). For a one-bit stage-2 model (G output u, Q2 = u * c+)
and a G checkpoint, run with the stage-2 protocol (as eval_s2_v11.run_q2 via figs_s2extra_v12.run: vision shown at
t = 0 only, process-1 motion visible, model.eval(), util.seed_all(0), cudnn deterministic) on the pref test (main)
and on v11_s2t_green (as v12), and store per episode / step u. Measures:
  * u agreement: per episode the fraction of t = 1..99 with sign(u_t) == correct sign (v14-A Sec.3 C definition:
    +1 if A-2's liked colour is on bins 6,7); balanced accuracy over the 4 cells (layout x liked);
  * sign agreement (v12 Sec.39): over frames t = 1..99 x bins 0,1,6,7 where P_A2 != 0, sign(Q2) == sign(P_A2), per layout;
  * share of u > 0 per cell, u time series per cell (mean), u switching (sign changes per episode).
-> data/result/v14/<out>/u_<cond>_s<s>_ep<e>.npz / .json. Judgement: aggregate_u_v14.py.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _HERE)
import s2run_v14  # noqa  (registers the v14 classes for load_v12)
from eval_trial_v12 import RES, load_v12  # noqa
from eval_s2_v11 import load_set, true_pleasure  # noqa
from figs_s2extra_v12 import run  # noqa
from eval_inverse_v14 import labels  # noqa

BINS = [0, 1, 6, 7]


def measures(Q, d, P, shape7):
    u = Q[:, :, 7] / shape7                                  # (N, 100)
    cs = labels(d)
    cl = d['layout'].astype(int) * 2 + d['a2_liked'].astype(int)
    agree_ep = (np.sign(u[:, 1:100]) == cs[:, None]).mean(1)
    r = {'u_agree_ba': float(np.mean([agree_ep[cl == k].mean() for k in np.unique(cl)])),
         'u_agree_raw': float(agree_ep.mean()),
         'u_pos_by_cell': {int(k): float((u[cl == k, 1:100] > 0).mean()) for k in np.unique(cl)},
         'u_mean_by_cell': {int(k): float(u[cl == k, 1:100].mean()) for k in np.unique(cl)},
         'u_sign_changes_per_ep': float((np.diff(np.sign(u[:, 1:100]), axis=1) != 0).sum(1).mean()),
         'u_abs_mean': float(np.abs(u[:, 1:100]).mean())}
    q = Q[:, 1:100][..., BINS]; p = P[:, 1:100][..., BINS]
    nz = p != 0
    ag = np.sign(q) == np.sign(p)
    lay = np.repeat(d['layout'][:, None], 99, 1)[..., None]
    r['sign_agree_all'] = float(ag[nz].mean())
    for L, nm in ((0, 'original'), (1, 'swap_rg')):
        mk = nz & (lay == L)
        r[f'sign_agree_{nm}'] = float(ag[mk].mean())
    return r, u, agree_ep, cl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='v14_s2u')
    ap.add_argument('--runs', nargs='+', required=True, help='cond:seed')
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--sets', nargs='+', default=['pref', 'green'])
    ap.add_argument('--out', default='stage2u/u')
    ap.add_argument('--tag_suffix', default='')
    a = ap.parse_args()
    out = os.path.join(RES, 'v14', a.out)
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in a.sets}
    P = {k: true_pleasure(k, D[k]) for k in a.sets}
    for spec in a.runs:
        c, s = spec.split(':')
        s = int(s)
        for e in a.epochs:
            m, _ = load_v12(f'{a.prefix}_pref_{c}_s{s}', s, e, dev)
            shape7 = float(m.q2_shape[7])
            res, sav = {}, {}
            for k in a.sets:
                Q, _ = run(m, D[k], dev)
                r, u, ag, cl = measures(Q, D[k], P[k], shape7)
                res[k] = r
                sav[f'{k}_u'], sav[f'{k}_agree_ep'], sav[f'{k}_cell'] = u.astype(np.float32), ag, cl
            tag = f'u_{c}_s{s}_ep{e}{a.tag_suffix}'
            json.dump(res, open(os.path.join(out, tag + '.json'), 'w'), indent=1)
            np.savez_compressed(os.path.join(out, tag + '.npz'), **sav)
            print(tag, json.dumps({k: {q: (round(v, 3) if isinstance(v, float) else v) for q, v in r.items()
                                       if q in ('u_agree_ba', 'sign_agree_original', 'sign_agree_swap_rg', 'u_pos_by_cell')}
                                   for k, r in res.items()}), flush=True)


if __name__ == '__main__':
    main()
