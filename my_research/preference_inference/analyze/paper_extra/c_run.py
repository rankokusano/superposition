"""
Paper extra C (docs/paper_extra_log.md Sec.4.3, 14.2): run the existing evaluation scripts on the sign-randomised
(paper_extra/C_signrand/s1_signrand_*) and zero-pleasure (paper_extra/C_pzero/s1_pzero_*) runs without changing
them. The signed-pleasure class is registered in model_v12 (as s2run_v14 does); evaluation data have no a1_liked_t,
so the sign is +1 (Red - Green, as the 80% condition). Outputs under data/result/paper_extra/C_signrand/eval/.
Usage (inside the container, from my_research/preference_inference):
    python analyze/paper_extra/c_run.py trial   <eval_trial_v12.py arguments>   (G1-G3, self error; --conds signrand|pzero)
    python analyze/paper_extra/c_run.py g5      <eval_g5_v13.py arguments>      (G5a, G5b)
    python analyze/paper_extra/c_run.py decomp  --seed <s> [--epochs ...]      (sign-following parts; conds signrand, pzero)
    python analyze/paper_extra/c_run.py d6      <a_cross_eval.py arguments>     (true-pleasure condition; conds signrand, pzero)
    python analyze/paper_extra/c_run.py inverse <eval_inverse_v14.py arguments> (specs csr|csrmse|cpz|cpzmse:<c>:<s>)
    python analyze/paper_extra/c_run.py g3flip  --models <cond>:<seed> ... [--epochs ...]
          G3 input-freeze (K = 15) with the pleasure held (hold), zero (zero) or held with the sign flipped (flip, -P(t0)),
          t < 20 canonical; g per episode as eval_trial_v12; d_P = (g_hold - g_flip) / 2. conds signrand, pzero,
          mixed80 (v13_s1_mixed_mse), random (v12_s1_random_mse). -> C_signrand/eval/g3flip/<cond>_s<s>_ep<e>.npz/.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_PI, 'analyze', 'v14'), os.path.join(_PI, 'analyze', 'v13'), os.path.join(_PI, 'analyze', 'v12'),
          os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)

from model import model_v12, model_v14  # noqa
if not hasattr(model_v12, 'SuperpositionNetworkMotionPleasureHoldSigned'):
    model_v12.SuperpositionNetworkMotionPleasureHoldSigned = model_v14.SuperpositionNetworkMotionPleasureHoldSigned
import eval_trial_v12  # noqa

RES = eval_trial_v12.RES
OUT = os.path.join(RES, 'paper_extra', 'C_signrand', 'eval')
MSE = {'signrand': 'paper_extra/C_signrand/s1_signrand_mse', 'pzero': 'paper_extra/C_pzero/s1_pzero_mse',
       'mixed80': 'v13_s1_mixed_mse', 'random': 'v12_s1_random_mse'}
L1 = {'signrand': 'paper_extra/C_signrand/s1_signrand_l1_s{s}', 'pzero': 'paper_extra/C_pzero/s1_pzero_l1_s{s}'}


@torch.no_grad()
def freeze_flip(model, p_mask, dev, K, mode, batch=50):
    """eval_trial_v12.freeze with the held pleasure P(t0) (hold), 0 (zero) or -P(t0) (flip)."""
    import h5py
    import util
    g = h5py.File(eval_trial_v12.COMMON, 'r')['test']
    N = g['self_vision'].shape[0]
    out = []
    for b0 in range(0, N, batch):
        x, m = eval_trial_v12.batch_x(g, b0, batch, dev)
        B = x.shape[0]
        util.seed_all(0)
        model.init_state(B)
        T0 = eval_trial_v12.T0
        for t in range(T0):
            pm = 0 if t == 0 else p_mask
            model({'self_vision': x[:, t], 'self_motion': m[:, t]}, pm, pm)
        P = {'hold': 1.0, 'flip': -1.0, 'zero': 0.0}[mode] * model.compute_pleasure(x[:, T0])
        hid = torch.ones(B, dtype=torch.bool, device=dev)
        preds = []
        for k in range(K):
            pr = model({'self_vision': x[:, T0 + k], 'self_motion': m[:, T0 + k]}, 1.0, 1.0, p1_override=P,
                       hidden_override=hid)['self_vision']
            preds.append(((pr + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
        out.append(np.stack(preds, 1))
    return np.concatenate(out)


def g3flip():
    import h5py
    from localise_v11 import dstar, slopes, g_of
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    args = ap.parse_args()
    out = os.path.join(OUT, 'g3flip')
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    fv = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_dist_views.h5'), 'r')
    views, vvalid = fv['views'][()], fv['valid'][()]
    K = 15
    for spec in args.models:
        c, s = spec.split(':')
        for e in args.epochs:
            model, p_mask = eval_trial_v12.load_v12(MSE[c], int(s), e, dev)
            per, r = {}, {'exp': MSE[c], 'seed': int(s), 'epoch': e}
            for mode in ('hold', 'zero', 'flip'):
                per[mode] = g_of(slopes(dstar(freeze_flip(model, p_mask, dev, K, mode), views, vvalid), K))
                r[mode] = eval_trial_v12.summ(per[mode])
            per['dP'] = (per['hold'] - per['flip']) / 2
            r['dP'] = eval_trial_v12.summ(per['dP'])
            np.savez_compressed(os.path.join(out, f'{c}_s{s}_ep{e}.npz'), **per)
            with open(os.path.join(out, f'{c}_s{s}_ep{e}.json'), 'w') as f:
                json.dump(r, f, indent=1)
            print(c, s, e, {q: round(v['mean'], 4) for q, v in r.items() if isinstance(v, dict)}, flush=True)


def main():
    which = sys.argv.pop(1)
    if which == 'g3flip':
        sys.argv[0] = __file__
        return g3flip()
    if which == 'trial':
        mod = eval_trial_v12
    elif which == 'g5':
        import eval_g5_v13 as mod
    elif which == 'decomp':
        import g5a_decomp_v13 as mod
        mod.EXPS = dict(L1)
        mod.RES = os.path.join(OUT, 'decomp')        # -> .../decomp/v13/post/g5a_decomp/s<s>
    elif which == 'd6':
        import a_cross_eval as mod
        mod.EXPS.update(L1)
        mod.OUT = os.path.join(OUT, 'd6')
    elif which == 'inverse':
        import eval_inverse_v14 as mod
        _load = mod.load

        def load(spec, epoch, dev):
            fam, c, s = spec.split(':')
            names = {'csr': L1['signrand'], 'cpz': L1['pzero'], 'csrmse': MSE['signrand'], 'cpzmse': MSE['pzero']}
            if fam in names:
                m, _ = eval_trial_v12.load_v12(names[fam].format(s=s), int(s), epoch, dev)
                m.motion_masking = False   # stage-2 protocol, as eval_inverse_v14.load
                return m, 11
            return _load(spec, epoch, dev)
        mod.load = load
    else:
        raise SystemExit(f'unknown {which}')
    sys.argv[0] = mod.__file__
    mod.main()


if __name__ == '__main__':
    main()
