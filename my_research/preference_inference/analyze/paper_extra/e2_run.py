"""
Paper round 11, E2 (docs/paper_extra_log.md Sec.35.2-35.3; instructions paper/check/paper_instructions_round11.md
Sec.2): does the correct preference help the prediction of the other more than the opposite one? No training.
A stage-1 model is run as in the matching (eval_inverse_v14.run, unchanged: vision shown at t = 0 only, masked p = 1
for t > 0 in both processes, process-1 motion never hidden, util.seed_all(0), model.eval(), cudnn deterministic), with
process-2 = [motion 0, candidate (8), flag 1] held for all steps, candidate =
  true : P_A2(t)   A-2's true pleasure (eval_s2_v11.true_pleasure; = the true-pleasure condition's input)
  opp  : -P_A2(t)  the pleasure A-2 would have with the opposite preference (exactly -P_A2; Sec.35.1)
  zero : 0
Per episode and step t = 0..99 (target frame t + 1) eval_inverse_v14.run gives L1 (self-vision |pred - x| per pixel,
[-1, 1] scale) and FP (other-side feature-prediction MSE). As in the matching, the MSE-stage points keep FP and the
L1-stage points keep L1 (FP does not change in the L1 stage; Sec.35.1).
Models (fam:seed): none v12_s1_random, p50 v12_s1_mixed, p80 v13_s1_mixed, sr80 paper_extra/C_signrand/s1_signrand,
sr50 paper_extra/E1_signrand50/s1_signrand50 (E1, after training). Signed classes get no a1_liked: sign +1 (Red).
-> data/result/paper_extra/E2/runs/<fam>_<stage>_s<s>_ep<e>_<set><tag>.npz
Usage (inside the container): python analyze/paper_extra/e2_run.py --models p50:0 none:0 [--stages mse l1] [--epochs ...]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_PI, 'analyze', 'v14'), os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)
import c_run  # noqa  (registers the signed class in model_v12, as for the signrand evaluations)
import a_cross_eval  # noqa  (registers green_a2p80 / pref_a2p80)
from eval_trial_v12 import RES, load_v12  # noqa
from eval_s2_v11 import load_set, true_pleasure  # noqa
import eval_inverse_v14  # noqa

OUT = os.path.join(RES, 'paper_extra', 'E2', 'runs')
EXPS = {'none': ('v12_s1_random_mse', 'v12_s1_random_l1_s{s}'),
        'p50': ('v12_s1_mixed_mse', 'v12_s1_mixed_l1_s{s}'),
        'p80': ('v13_s1_mixed_mse', 'v13_s1_mixed_l1_s{s}'),
        'sr80': (c_run.MSE['signrand'], c_run.L1['signrand']),
        'sr50': ('paper_extra/E1_signrand50/s1_signrand50_mse', 'paper_extra/E1_signrand50/s1_signrand50_l1_s{s}')}
KEEP = {'mse': 'FP', 'l1': 'L1'}
CANDS = ('true', 'opp', 'zero')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='fam:seed, fam in ' + '|'.join(EXPS))
    ap.add_argument('--stages', nargs='+', default=['mse', 'l1'])
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--sets', nargs='+', default=['pref', 'green', 'pref_a2p80', 'green_a2p80'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in args.sets}
    P = {k: true_pleasure(k, D[k]) for k in args.sets}
    for spec in args.models:
        fam, s = spec.split(':')
        for st in args.stages:
            exp = EXPS[fam][0 if st == 'mse' else 1].format(s=s)
            for e in args.epochs:
                t0 = time.time()
                model, _ = load_v12(exp, int(s), e, dev)
                model.motion_masking = False   # stage-2 protocol: process-1 motion visible (flag 0)
                for k in args.sets:
                    N = P[k].shape[0]
                    res = {c: eval_inverse_v14.run(model, 11, D[k], x, dev, N)
                           for c, x in (('true', P[k]), ('opp', -P[k]), ('zero', 0 * P[k]))}
                    q = KEEP[st]
                    sav = {f'{q}_{c}': res[c][q] for c in CANDS}
                    sav.update(a2_liked=D[k]['a2_liked'][:N], layout=D[k]['layout'][:N])
                    tag = f'{fam}_{st}_s{s}_ep{e}_{k}{args.tag}'
                    np.savez_compressed(os.path.join(OUT, f'{tag}.npz'), **sav)
                    u = sav[f'{q}_opp'][:, :50].mean(1) - sav[f'{q}_true'][:, :50].mean(1)
                    print(tag, exp, q, 'U_pref t1-50 mean', round(float(u.mean()), 6),
                          {c: round(float(sav[f'{q}_{c}'].mean()), 6) for c in CANDS}, f'{time.time() - t0:.1f}s', flush=True)
                del model
                torch.cuda.empty_cache()
    print(json.dumps({'done': args.models, 'stages': args.stages, 'epochs': args.epochs, 'sets': args.sets, 'tag': args.tag}))


if __name__ == '__main__':
    main()
