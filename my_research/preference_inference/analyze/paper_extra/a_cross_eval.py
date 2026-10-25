"""
Paper extra A (docs/paper_instructions_extra.md Sec.2; pre-registration docs/paper_extra_log.md): the true-pleasure
condition of a stage-1 model on stage-2 test sets whose A-2 follows its preference with p = 0.5 (existing sets) or
p = 0.8 (new sets). Same run as analyze/v12/eval_d6_v12.run (vision shown at t = 0 only, masked p = 1 for t > 0 in
both processes; process-1 motion never hidden; util.seed_all(0); model.eval()), with process-2 =
  zero : [motion 0, pleasure 0,    flag 1]   (eval_d6_v12 'A', T3 'zero')
  true : [motion 0, pleasure P_A2, flag 1]   (eval_d6_v12 'B', T3 'd6')
P_A2 is A-2's true pleasure from A-2's own view (eval_s2_v11.true_pleasure). Per episode and step t = 0..99
(target frame t + 1) it stores
  fp    : other-side feature-prediction MSE, predict_feature(h2_t) vs Encoder-2(x_{t+1})
  self  : self-vision L1 per pixel on the [0, 1] scale (as eval_d6_v12)
  a2err : A-2 position error read from the predicted image (p2test_v11.read_a2; calibration on the ORIGINAL
          v11_s2t_green true views for every set, as eval_d6_v12), NaN where not read
-> data/result/paper_extra/A_cross/<out>/<tag>_<set>.npz (+ .json with the means).
Models: random = v12_s1_random_l1_s<s>, mixed50 = v12_s1_mixed_l1_s<s>, mixed80 = v13_s1_mixed_l1_s<s>.
Run inside the container from my_research/preference_inference, e.g.
    python analyze/paper_extra/a_cross_eval.py --models mixed80:1 --sets green --device cpu --threads 4
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _PI)
from eval_trial_v12 import RES, load_v12  # noqa
import eval_s2_v11  # noqa
from eval_s2_v11 import load_set, true_pleasure  # noqa
from p2test_v11 import calibrate, read_a2  # noqa
import util  # noqa

EXPS = {'random': 'v12_s1_random_l1_s{s}', 'mixed50': 'v12_s1_mixed_l1_s{s}', 'mixed80': 'v13_s1_mixed_l1_s{s}'}
# p = 0.8 versions of the stage-2 test sets (paper_extra_log Sec.A data); the p = 0.5 sets are eval_s2_v11.SETS
eval_s2_v11.SETS.update({'green_a2p80': ('paper_s2t_green_a2p80', 'test'), 'pref_a2p80': ('paper_s2_pref_a2p80', 'test')})
OUT = os.path.join(RES, 'paper_extra', 'A_cross')


@torch.no_grad()
def run(model, d, P8, dev, alpha, beta, n_ep, batch=50):
    g = h5py.File(d['path'], 'r')[d['split']]
    util.seed_all(0)
    FP, SL, E2 = [], [], []
    for b0 in range(0, n_ep, batch):
        b1 = min(b0 + batch, n_ep)
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b1]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b1]).float().to(dev)
        sp, op = d['self_position'][b0:b1], d['other_position'][b0:b1]
        B = x.shape[0]
        model.init_state(B)
        fp, sl, e2 = [], [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            q = torch.zeros(B, 11, device=dev)
            q[:, 10] = 1.0
            if P8 is not None:
                q[:, 2:10] = torch.from_numpy(P8[b0:b1, t]).float().to(dev)
            pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=q)['self_vision']
            p01 = ((pr + 1) / 2).clamp(0, 1)
            sl.append((p01 - (x[:, t + 1] + 1) / 2).abs().mean((1, 2, 3)).cpu().numpy())
            est = read_a2(p01.permute(0, 2, 3, 1).cpu().numpy(), sp[:, t + 1], alpha, beta)
            e2.append(np.linalg.norm(est - op[:, t + 1], axis=1))
            os_ = model.get_state()['other'][0]
            fp.append(F.mse_loss(model.predict_feature(os_), model.other_vision_encoder_module(x[:, t + 1]),
                                 reduction='none').mean(1).cpu().numpy())
        FP.append(np.stack(fp, 1)); SL.append(np.stack(sl, 1)); E2.append(np.stack(e2, 1))
    return {'fp': np.concatenate(FP), 'self': np.concatenate(SL), 'a2err': np.concatenate(E2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='cond:seed, cond in random|mixed50|mixed80')
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--sets', nargs='+', default=['green', 'pref', 'green_a2p80', 'pref_a2p80'])
    ap.add_argument('--n_ep', type=int, default=0, help='0 = all test episodes')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--out', default='eval')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    out = os.path.join(OUT, args.out)
    os.makedirs(out, exist_ok=True)
    D = {k: load_set(k) for k in args.sets}
    PA2 = {k: true_pleasure(k, D[k]) for k in args.sets}
    dg = load_set('green')
    gg = h5py.File(dg['path'], 'r')['test']
    alpha, beta = calibrate(gg['self_vision'][:, :100:5], dg['self_position'][:, :100:5], dg['other_position'][:, :100:5])
    for spec in args.models:
        c, s = spec.split(':')
        exp = EXPS[c].format(s=s)
        model, _ = load_v12(exp, int(s), args.epoch, dev)
        model.motion_masking = False   # stage-2 protocol: process-1 motion visible (flag 0)
        for k in args.sets:
            n_ep = args.n_ep or PA2[k].shape[0]
            t0 = time.time()
            r = {f'{v}_{q}': a for v, P8 in (('zero', None), ('true', PA2[k])) for q, a in run(model, D[k], P8, dev, alpha, beta, n_ep).items()}
            r['a2_source'] = D[k]['a2_source'][:n_ep, :100]
            tag = f'{c}_s{s}_ep{args.epoch}_{k}'
            np.savez_compressed(os.path.join(out, f'{tag}.npz'), **r)
            summ = {'exp': exp, 'epoch': args.epoch, 'set': k, 'n_ep': n_ep, 'device': args.device,
                    'threads': args.threads, 'seconds': round(time.time() - t0, 1), 'calib': [alpha, beta]}
            for v in ('zero', 'true'):
                summ[v] = {'fp_t1_99': float(r[f'{v}_fp'][:, 1:].mean()), 'fp_t0_99': float(r[f'{v}_fp'].mean()),
                           'self_t1_99': float(r[f'{v}_self'][:, 1:].mean()),
                           'a2err_t1_99': float(np.nanmean(np.nanmean(r[f'{v}_a2err'][:, 1:], 1)))}
            summ['delta_fp_t1_99'] = summ['true']['fp_t1_99'] - summ['zero']['fp_t1_99']
            with open(os.path.join(out, f'{tag}.json'), 'w') as f:
                json.dump(summ, f, indent=1)
            print(tag, json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
