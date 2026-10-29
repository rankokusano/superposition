"""
Paper round 11, E2 addition (docs/paper_extra_log.md Sec.35.8; outside the pre-registration, procedure committed before
computing): A-2 position error read from the predicted image, and the red-form bias with the stage-1 sign convention.
Same run as e2_run.py (eval_inverse_v14.run unchanged; process-2 = [0, candidate, 1] held for all steps, candidate
true = P_A2, opp = -P_A2, zero), with keep_pred=True on the L1-stage points: the A-2 position is read from each
predicted frame by p2test_v11.read_a2 (NaN where the black mass < MIN_MASS, as before; calibration p2test_v11.calibrate
on the original green test set, as eval_d6_v12 / a_cross_eval / eval_inverse_v14 --keep_pred), and
  dev[t] = |read position - true A-2 position (other_position[t + 1])|   (target frame t + 1; stored for t = 0..49)
Families: e2_run.EXPS (none, p50, p80, sr80, sr50) plus v14-C v14c (v14_s1_mixedpref, A-1's liked colour per episode,
P = liked - disliked) and v14cr (v14_s1_randompref, its no-experience version).
--a1 green flips the sign of process-1's pleasure at evaluation only (as eval_inverse_v14 --a1_liked green).
Stored per file: dev_<c> (N, 50) for the L1 stage; L1_<c>, FP_<c> (N, 100) for v14c / v14cr (any stage) and for the
MSE stage; for the e2 families the L1 / FP arrays are compared on the fly with the E2 runs instead
(chk_L1_vs_e2: this run's L1 vs E2's L1-stage file of the same point; chk_FP_vs_e2mse200: this run's FP vs E2's
MSE-stage ep200 FP, which the L1 stage does not change).
-> data/result/paper_extra/E2/pos/runs/<fam>_<stage>_s<s>_ep<e>_<set>[_a1green]<tag>.npz
Usage (inside the container): python analyze/paper_extra/e2pos_run.py --models p50:0 none:0 [--stages l1] [--a1 red]
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_PI, 'analyze', 'v14'), os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)
import e2_run  # noqa  (also registers the signed class and the A-2 0.8 sets)
import s2run_v14  # noqa  (registers the v14 classes for load_v12)
from eval_trial_v12 import RES, load_v12  # noqa
from eval_s2_v11 import load_set, true_pleasure  # noqa
from p2test_v11 import calibrate  # noqa
import eval_inverse_v14  # noqa

OUT = os.path.join(RES, 'paper_extra', 'E2', 'pos', 'runs')
E2RUNS = os.path.join(RES, 'paper_extra', 'E2', 'runs')
EXPS = dict(e2_run.EXPS, v14c=('v14_s1_mixedpref_mse', 'v14_s1_mixedpref_l1_s{s}'),
            v14cr=('v14_s1_randompref_mse', 'v14_s1_randompref_l1_s{s}'))
CANDS = ('true', 'opp', 'zero')
TW = 50   # stored steps t = 0..49 (target frames 1-50; the window 10-50 is t = 9..49)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='fam:seed, fam in ' + '|'.join(EXPS))
    ap.add_argument('--stages', nargs='+', default=['l1'])
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--sets', nargs='+', default=['pref', 'green', 'pref_a2p80', 'green_a2p80'])
    ap.add_argument('--a1', default='red', choices=['red', 'green'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in args.sets}
    P = {k: true_pleasure(k, D[k]) for k in args.sets}
    dg = load_set('green')
    gg = h5py.File(dg['path'], 'r')['test']
    ab = calibrate(gg['self_vision'][:, :100:5], dg['self_position'][:, :100:5], dg['other_position'][:, :100:5])
    for spec in args.models:
        fam, s = spec.split(':')
        for st in args.stages:
            exp = EXPS[fam][0 if st == 'mse' else 1].format(s=s)
            for e in args.epochs:
                t0 = time.time()
                model, _ = load_v12(exp, int(s), e, dev)
                model.motion_masking = False   # stage-2 protocol: process-1 motion visible (flag 0)
                assert model.liked == 'Red' and model.disliked == 'Green', (model.liked, model.disliked)
                if args.a1 == 'green':         # A-1 likes Green at evaluation only: P = Green - Red
                    _orig = model.compute_pleasure
                    model.compute_pleasure = lambda sv, _f=_orig: -_f(sv)
                for k in args.sets:
                    d, N = D[k], P[k].shape[0]
                    keep = st == 'l1'
                    res = {c: eval_inverse_v14.run(model, 11, d, x, dev, N, keep_pred=keep, ab=ab if keep else None)
                           for c, x in (('true', P[k]), ('opp', -P[k]), ('zero', 0 * P[k]))}
                    sav = {'a2_liked': d['a2_liked'][:N], 'layout': d['layout'][:N], 'ab': np.array(ab)}
                    if keep:
                        op = d['other_position'][:N, 1:TW + 1]
                        for c in CANDS:
                            sav[f'dev_{c}'] = np.linalg.norm(res[c]['est'][:, :TW] - op, axis=-1).astype(np.float32)
                    if fam in ('v14c', 'v14cr') or st == 'mse':
                        for c in CANDS:
                            sav[f'L1_{c}'], sav[f'FP_{c}'] = res[c]['L1'], res[c]['FP']
                    f_l1 = os.path.join(E2RUNS, f'{fam}_l1_s{s}_ep{e}_{k}.npz')
                    f_mse = os.path.join(E2RUNS, f'{fam}_mse_s{s}_ep200_{k}.npz')
                    if st == 'l1' and args.a1 == 'red' and os.path.exists(f_l1):
                        z = np.load(f_l1)
                        sav['chk_L1_vs_e2'] = np.array([np.array_equal(res[c]['L1'], z[f'L1_{c}']) for c in CANDS])
                        sav['chk_L1_vs_e2_max'] = np.array(max(float(np.abs(res[c]['L1'] - z[f'L1_{c}']).max()) for c in CANDS))
                    if st == 'l1' and os.path.exists(f_mse):
                        z = np.load(f_mse)
                        sav['chk_FP_vs_e2mse200'] = np.array([np.array_equal(res[c]['FP'], z[f'FP_{c}']) for c in CANDS])
                    tag = f'{fam}_{st}_s{s}_ep{e}_{k}' + ('_a1green' if args.a1 == 'green' else '') + args.tag
                    np.savez_compressed(os.path.join(OUT, f'{tag}.npz'), **sav)
                    msg = {x: sav[x].tolist() for x in sav if x.startswith('chk')}
                    if keep:
                        w = slice(9, TW)
                        msg['frac_read'] = {c: round(float(np.isfinite(sav[f'dev_{c}'][:, w]).mean()), 3) for c in CANDS}
                        msg['dev_mean'] = {c: round(float(np.nanmean(sav[f'dev_{c}'][:, w])), 4) for c in CANDS}
                    print(tag, exp, json.dumps(msg), f'{time.time() - t0:.1f}s', flush=True)
                del model
                torch.cuda.empty_cache()
    print(json.dumps({'done': args.models, 'stages': args.stages, 'epochs': args.epochs, 'sets': args.sets,
                      'a1': args.a1, 'tag': args.tag}))


if __name__ == '__main__':
    main()
