"""
v12: the D6-like gate (docs/v12_experiment_log.md Sec.10, item 3). A stage-1 model is run like stage 2
(vision shown at t = 0 only, masked p = 1 for t > 0 in both processes; process-1 motion never hidden,
flag 0, current pleasure; util.seed_all(0)) on v11_s2t_green and the v11_s2_pref test split, with
process-2 =
  A: [motion 0, pleasure 0,    flag 1]
  B: [motion 0, pleasure P_A2, flag 1]   (A-2's true pleasure, v11 Sec.7.42 definition)
  C: all 11 zeros (as in stage 1; reference)
Per episode: A-2 position error read from the predicted image (v11 Sec.7.53 read-out, distance
calibrated on v11_s2t_green true views), other-side feature-prediction MSE, self-vision L1; and
h2 -> A-2 position R^2. Main quantity A - B (positive = P_A2 helps). Writes
data/result/v12/<sub>/d6_<cond>_ep<e>.json and per-episode npz; with both conditions also the paired file.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_trial_v12 import RES, boot_ci, load_v12, summ  # noqa
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_s2_v11 import load_set, true_pleasure, r2_cv  # noqa
from p2test_v11 import calibrate, read_a2  # noqa
import util  # noqa

SETS = ('green', 'pref')


@torch.no_grad()
def run(model, d, P2, dev, alpha, beta, batch=50):
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    util.seed_all(0)
    E2, FP, SL, H2 = [], [], [], []
    for b0 in range(0, N, batch):
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b0 + batch]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b0 + batch]).float().to(dev)
        sp = d['self_position'][b0:b0 + batch]
        op = d['other_position'][b0:b0 + batch]
        model.init_state(x.shape[0])
        e2, fp, sl, h2 = [], [], [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            p2 = None if P2 is None else torch.from_numpy(P2[b0:b0 + batch, t]).float().to(dev)
            pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=p2)['self_vision']
            p01 = ((pr + 1) / 2).clamp(0, 1)
            sl.append((p01 - (x[:, t + 1] + 1) / 2).abs().mean((1, 2, 3)).cpu().numpy())
            est = read_a2(p01.permute(0, 2, 3, 1).cpu().numpy(), sp[:, t + 1], alpha, beta)
            e2.append(np.linalg.norm(est - op[:, t + 1], axis=1))
            os_ = model.get_state()['other'][0]
            fp.append(F.mse_loss(model.predict_feature(os_), model.other_vision_encoder_module(x[:, t + 1]),
                                 reduction='none').mean(1).cpu().numpy())
            h2.append(os_.cpu().numpy())
        E2.append(np.stack(e2, 1)); FP.append(np.stack(fp, 1)); SL.append(np.stack(sl, 1)); H2.append(np.stack(h2, 1))
    E2, FP, SL, H2 = (np.concatenate(a) for a in (E2, FP, SL, H2))
    ts = np.arange(1, 100)
    Y = d['other_position'][:, ts].reshape(-1, 2)
    grp = np.repeat(np.arange(N)[:, None], len(ts), 1).reshape(-1)
    return {'a2err_ep': np.nanmean(E2[:, 1:], 1), 'a2_frac_read': float(np.isfinite(E2[:, 1:]).mean()),
            'fp_ep': FP[:, 1:].mean(1), 'self_ep': SL[:, 1:].mean(1),
            'h2_A2pos': r2_cv(H2[:, ts].reshape(-1, 128), Y, grp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', default='v12_trial_pc_{}_mse')
    ap.add_argument('--conds', nargs='+', default=['random', 'mixed'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epoch', type=int, default=150)
    ap.add_argument('--sub', default='trial/d6_pc')
    args = ap.parse_args()
    out = os.path.join(RES, 'v12', args.sub)
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    D = {k: load_set(k) for k in SETS}
    PA2 = {k: true_pleasure(k, D[k]) for k in SETS}
    g = h5py.File(D['green']['path'], 'r')['test']
    alpha, beta = calibrate(g['self_vision'][:, :100:5], D['green']['self_position'][:, :100:5], D['green']['other_position'][:, :100:5])
    per = {}
    for c in args.conds:
        model, _ = load_v12(args.exp_fmt.format(c), args.seed, args.epoch, dev)
        model.motion_masking = False   # stage-2 protocol: process-1 motion visible (flag 0)
        r = {'exp': args.exp_fmt.format(c), 'seed': args.seed, 'epoch': args.epoch, 'calib': [alpha, beta]}
        for k in SETS:
            N = PA2[k].shape[0]
            pA = np.zeros((N, 100, 11), np.float32); pA[..., 10] = 1
            pB = pA.copy(); pB[..., 2:10] = PA2[k][:, :100]
            res = {nm: run(model, D[k], P2, dev, alpha, beta) for nm, P2 in (('A', pA), ('B', pB), ('C', None))}
            rk = {}
            for nm, v in res.items():
                rk[nm] = {'a2err': float(np.nanmean(v['a2err_ep'])), 'a2_frac_read': v['a2_frac_read'],
                          'fp': float(v['fp_ep'].mean()), 'self_L1': float(v['self_ep'].mean()), 'h2_A2pos': v['h2_A2pos']}
            for q in ('a2err', 'fp', 'self'):
                dd = res['A'][f'{q}_ep'] - res['B'][f'{q}_ep']
                rk[f'A_minus_B_{q}'] = summ(dd)
                per[(c, k, q)] = dd
            r[k] = rk
        with open(os.path.join(out, f'd6_{c}_ep{args.epoch}.json'), 'w') as f:
            json.dump(r, f, indent=1)
        np.savez_compressed(os.path.join(out, f'd6_{c}_ep{args.epoch}_per.npz'),
                            **{f'{k}_{q}': per[(c, k, q)] for k in SETS for q in ('a2err', 'fp', 'self')})
        print(c, json.dumps({k: {q: r[k][f'A_minus_B_{q}']['mean'] for q in ('a2err', 'fp', 'self')} for k in SETS}), flush=True)
    if len(args.conds) == 2:
        paired = {f'{k}_{q}': summ(per[('mixed', k, q)] - per[('random', k, q)]) for k in SETS for q in ('a2err', 'fp', 'self')}
        with open(os.path.join(out, f'd6_paired_ep{args.epoch}.json'), 'w') as f:
            json.dump(paired, f, indent=1)
        print('paired', json.dumps({k: round(v['mean'], 4) for k, v in paired.items()}), flush=True)


if __name__ == '__main__':
    main()
