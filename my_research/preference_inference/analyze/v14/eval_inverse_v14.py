"""
v14-A inverse inference (docs/v14_instructions.md Sec.1.1, 3.1; pre-registration docs/v14_experiment_log.md).
A stage-1 model (no module G, no training) is run like stage 2 (vision shown at t = 0 only, masked p = 1 for
t > 0 in both processes; process-1 motion never hidden, flag 0, current pleasure; model.eval(),
util.seed_all(0), cudnn deterministic) on a stage-2 test set, once per process-2 candidate held for all steps:
  cp  : c(+1)  (bins 6,7 pleasant)          cm : c(-1)  (bins 0,1 pleasant)
  c0  : zeros (no preference)
  ref_red / ref_green : A-2's true pleasure computed from A-2's view as if A-2 liked Red / Green
        (uses A-2's position: reference only, never a claim)
v12/v13 models get process-2 = [motion 0, candidate (8), flag 1]; v11 models (8-dim process-2) get the candidate.
Per episode and step t = 0..99 (target frame t+1) it stores
  L1 : self-vision L1 per pixel, |pred - x_{t+1}| on the [-1, 1] scale (as eval_s2_v11.run_q2)
  FP : other-side feature-prediction MSE, predict_feature(h2) vs other encoder of x_{t+1} (as t1t3_s2_v11)
-> data/result/v14/<out>/<tag>_<set>.npz (+ labels). With --keep_pred also the predicted images and the
A-2 position read from them (v11 Sec.7.53 read-out) for cp / cm (smoke test and example figures).
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
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
sys.path.insert(0, _PI)
from eval_trial_v12 import RES, load_v12  # noqa
import eval_trial_v11  # noqa
from eval_s2_v11 import load_set  # noqa
from p2test_v11 import calibrate, read_a2  # noqa
import util  # noqa
from v11core.pleasure import pleasure_np  # noqa

OUT = os.path.join(RES, 'v14')
CANDS = ('cp', 'cm', 'c0', 'ref_red', 'ref_green')


def load(spec, epoch, dev):
    fam, c, s = spec.split(':')
    s = int(s)
    if fam in ('v12', 'v13'):
        m, _ = load_v12(f'{fam}_s1_{c}_l1_s{s}', s, epoch, dev)
        m.motion_masking = False   # stage-2 protocol: process-1 motion visible (flag 0)
        return m, 11
    if fam == 'v11':
        m, _ = eval_trial_v11.load_model(f'v11_s1_{c}_l1_s{s}', s, epoch, dev)
        return m, 8
    raise ValueError(spec)


def labels(d):
    """correct sign of u per episode: +1 if A-2's liked colour sits on bins 6,7 (layout 0 & Red, layout 1 & Green)."""
    lay, lk = d['layout'].astype(int), d['a2_liked'].astype(int)   # a2_liked: 0 Green, 1 Red
    return np.where(((lay == 0) & (lk == 1)) | ((lay == 1) & (lk == 0)), 1, -1)


def pa2_as(d, liked):
    ov = h5py.File(d['path'], 'r')[d['split']]['other_vision'][:, :100]
    return pleasure_np(ov, liked, 'Green' if liked == 'Red' else 'Red')


@torch.no_grad()
def run(model, width, d, P8, dev, n_ep, keep_pred=False, ab=None, batch=50):
    """P8: (N, 100, 8) candidate per episode and step."""
    g = h5py.File(d['path'], 'r')[d['split']]
    util.seed_all(0)
    L1, FP, PRED, EST = [], [], [], []
    for b0 in range(0, n_ep, batch):
        b1 = min(b0 + batch, n_ep)
        x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][b0:b1]))).to(dev)
        m = torch.from_numpy(g['self_motion'][b0:b1]).float().to(dev)
        B = x.shape[0]
        model.init_state(B)
        l1, fp, pr_, es = [], [], [], []
        for t in range(100):
            pm = 0 if t == 0 else 1
            c = torch.from_numpy(P8[b0:b1, t]).float().to(dev)
            if width == 11:
                z = torch.zeros(B, 1, device=dev)
                p2 = torch.cat([z, z, c, torch.ones_like(z)], 1)
            else:
                p2 = c
            pr = model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm, p2_override=p2)['self_vision']
            l1.append(torch.abs(pr - x[:, t + 1]).mean((1, 2, 3)).cpu().numpy())
            os_ = model.get_state()['other'][0]
            fp.append(F.mse_loss(model.predict_feature(os_), model.other_vision_encoder_module(x[:, t + 1]),
                                 reduction='none').mean(1).cpu().numpy())
            if keep_pred:
                p01 = ((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
                pr_.append(p01.astype(np.float16))
                es.append(read_a2(p01, d['self_position'][b0:b1, t + 1], *ab))
        L1.append(np.stack(l1, 1)); FP.append(np.stack(fp, 1))
        if keep_pred:
            PRED.append(np.stack(pr_, 1)); EST.append(np.stack(es, 1))
    out = {'L1': np.concatenate(L1), 'FP': np.concatenate(FP)}
    if keep_pred:
        out['pred'], out['est'] = np.concatenate(PRED), np.concatenate(EST)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='fam:cond:seed, e.g. v12:mixed:0')
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--sets', nargs='+', default=['pref', 'green'])
    ap.add_argument('--n_ep', type=int, default=0, help='0 = all test episodes')
    ap.add_argument('--out', default='inverse')
    ap.add_argument('--tag_suffix', default='')
    ap.add_argument('--keep_pred', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    out = os.path.join(OUT, args.out)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    shape = np.array(json.load(open(os.path.join(OUT, 'cand_shape.json')))['c_plus'], np.float32)
    D = {k: load_set(k) for k in args.sets}
    ab = None
    if args.keep_pred:
        gg = h5py.File(load_set('green')['path'], 'r')['test']
        dg = load_set('green')
        ab = calibrate(gg['self_vision'][:, :100:5], dg['self_position'][:, :100:5], dg['other_position'][:, :100:5])
    P = {}
    for k in args.sets:
        d = D[k]
        assert (d['layout'][:, None] == h5py.File(d['path'], 'r')[d['split']]['layout_t'][()]).all(), 'layout changes'
        N = len(d['layout']) if not args.n_ep else min(args.n_ep, len(d['layout']))
        one = np.ones((N, 100, 1), np.float32)
        P[k] = {'cp': one * shape, 'cm': -one * shape, 'c0': 0 * one * shape,
                'ref_red': pa2_as(d, 'Red')[:N], 'ref_green': pa2_as(d, 'Green')[:N]}
    for spec in args.models:
        for e in args.epochs:
            model, width = load(spec, e, dev)
            tag = spec.replace(':', '_') + f'_ep{e}' + args.tag_suffix
            for k in args.sets:
                d = D[k]
                N = P[k]['cp'].shape[0]
                res = {c: run(model, width, d, P[k][c], dev, N, args.keep_pred and c in ('cp', 'cm'), ab) for c in CANDS}
                sav = {f'{q}_{c}': res[c][q] for c in CANDS for q in ('L1', 'FP')}
                if args.keep_pred:
                    for c in ('cp', 'cm'):
                        sav[f'pred_{c}'], sav[f'est_{c}'] = res[c]['pred'], res[c]['est']
                    sav['ab'] = np.array(ab)
                sav.update(layout=d['layout'][:N], a2_liked=d['a2_liked'][:N], correct_sign=labels(d)[:N],
                           self_position=d['self_position'][:N], other_position=d['other_position'][:N])
                np.savez_compressed(os.path.join(out, f'{tag}_{k}.npz'), **sav)
                cs = sav['correct_sign']
                dl = res['cm']['L1'][:, :50].sum(1) - res['cp']['L1'][:, :50].sum(1)   # > 0: cp better
                acc = ((dl > 0) == (cs > 0)).mean()
                print(tag, k, 'N', N, 'raw acc T50 (L1)', round(float(acc), 3),
                      'mean L1', {c: round(float(res[c]['L1'].mean()), 5) for c in CANDS}, flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
