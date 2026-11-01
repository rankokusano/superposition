"""
Paper round 17 Sec.1 (docs/paper_instructions_round17.md; log Sec.41): checks before the computation. No training.
  --vars       behaviour variables of the stage-B set (round17_feat.variables): shapes, finiteness, ranges, rest
               fraction; and how A-2's stored motion relates to its displacement (descriptive).
  --models ... (GPU) per checkpoint: round 16's (pos) rebuilt with the same code (round16_eval.pos_fit on the saved
               stage-C Q2, which is bit-identical to round 16's G output): R^2 vs the round-16 json; the loss with the
               rebuilt (pos) vs the round-16 saved losses; on the first model also the losses with G's own output
               (a) and with 0 (z) vs round 16 (to reuse them).
  --summary    collects the per-checkpoint checks and the 50% S_pos / R^2 (4 pairs x 5 points) from the rebuilt losses.
-> data/result/paper_extra/Round17/precheck/
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import round17_feat as F17  # noqa
R16 = F17.R16
C = R16.C
OUT = os.path.join(C.RES, 'paper_extra', 'Round17', 'precheck')
EV16 = os.path.join(C.RES, 'paper_extra', 'Round16', 'eval')


def check_vars():
    X, P2, P1, info = F17.variables()
    import h5py
    d = C.E.load_set('vdirb')
    g = h5py.File(d['path'], 'r')[d['split']]
    om = g['other_motion'][()].astype(np.float64)[:, :100]
    src = g['a2_source'][()][:, :100]
    P = g['other_position'][()].astype(np.float64)
    dP = P[:, 1:101] - P[:, :100]
    n = np.linalg.norm(dP, axis=-1)
    mv = n > 1e-6
    cosang = (om * dP).sum(-1)[mv] / (np.linalg.norm(om, axis=-1)[mv] * n[mv])
    R = dict(info, shapes={k: list(v.shape) for k, v in X.items()},
             finite={k: bool(np.isfinite(v).all()) for k, v in X.items()},
             ranges={k: [float(v.min()), float(v.max())] for k, v in X.items()},
             step_len={'moving_frac': float(mv.mean()), 'len_median': float(np.median(n[mv])), 'len_max': float(n.max()),
                       'still_steps_rest': int(((~mv) & (src == 3)).sum()), 'still_steps_not_rest': int(((~mv) & (src != 3)).sum())},
             motion_vs_disp_cos={'median': float(np.median(cosang)), 'frac_gt_0.99': float((cosang > 0.99).mean())})
    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(os.path.join(OUT, 'vars.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1), flush=True)


def check_model(exp, s, e, dev, P2, P1, first):
    import h5py
    import torch
    sv = np.load(os.path.join(C.OUT, 'eval', f'{exp}_ep{e}.npz'))
    Qa = sv['Q2']
    j16 = json.load(open(os.path.join(EV16, f'{exp}_ep{e}.json')))
    z16 = np.load(os.path.join(EV16, f'{exp}_ep{e}.npz'))
    qp, r2 = R16.pos_fit(Qa, P2, P1)
    R = {'model': f'{exp} ep{e}', 'r2': r2, 'r2_16': j16['r2_pos'],
         'r2_maxabs': max(abs(r2[w]['pooled'] - j16['r2_pos'][w]['pooled']) for w in r2)}
    model, _ = C.load_v12(exp, s, e, dev)
    model.eval()
    d = C.E.load_set('vdirb')
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    q, R['clip_pos'] = R16.clip(qp)
    R['clip_pos_16'] = j16['clip']['pos']
    ins = {'pos': q}
    if first:
        ins['a'] = None
        ins['z'] = np.zeros_like(Qa)
    for k, Q in ins.items():
        if Q is None:
            L, _, _ = R16.run(model, g, N, dev, None, C.Hook(model.motion_generator_module.fc))
        else:
            L = R16.run(model, g, N, dev, Q)
        for w, sl in R16.WIN.items():
            for i, nm in enumerate(('L1', 'FP1', 'FP2')):
                a = L[:, sl, i].mean(1).astype(np.float32)
                b = z16[f'{k}|{w}|{nm}']
                R[f'{k}|{w}|{nm}|identical'] = bool(np.array_equal(a, b))
                R[f'{k}|{w}|{nm}|maxabs'] = float(np.abs(a - b).max())
    return R


def summary():
    rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, 'models', '*.json')))]
    R = {'n': len(rs), 'r2_maxabs': max(r['r2_maxabs'] for r in rs),
         'pos_losses_identical_all': all(v for r in rs for k, v in r.items() if k.startswith('pos|') and k.endswith('identical')),
         'pos_losses_maxabs': max(v for r in rs for k, v in r.items() if k.startswith('pos|') and k.endswith('maxabs')),
         'clip_equal_all': all(r['clip_pos'] == r['clip_pos_16'] for r in rs),
         'a_z_identical': {k: v for r in rs for k, v in r.items() if k[:2] in ('a|', 'z|') and k.endswith('identical')}}
    # 50% S_pos and R^2 from the rebuilt checks (identical losses -> the round-16 losses give the same S_pos)
    S16 = json.load(open(os.path.join(C.RES, 'paper_extra', 'Round16', 'analysis', 'summary.json')))
    R['S_pos_p50_round16'] = S16['main']['p50']['tot|S_pos']['mean']
    r2p = [r['r2']['0_49']['pooled'] for r in rs if 'pref_mixed_s' in r['model'] and 'v14' not in r['model']]
    R['r2_p50_mean_rebuilt'] = float(np.mean(r2p)) if r2p else None
    json.dump(R, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    print(json.dumps(R, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vars', action='store_true')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--models', nargs='*', default=[])
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, 'models'), exist_ok=True)
    if a.vars:
        return check_vars()
    if a.summary:
        return summary()
    import torch
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    P2, P1 = R16.load_pos()
    first = True
    for spec in a.models:
        exp, s, eps = spec.split(':')
        for e in map(int, eps.split(',')):
            R = check_model(exp, int(s), e, dev, P2, P1, first and spec.startswith('v12_s2_pref_mixed_s0'))
            first = False
            json.dump(R, open(os.path.join(OUT, 'models', f'{exp}_ep{e}.json'), 'w'), indent=1)
            print(R['model'], 'r2 diff', R['r2_maxabs'], 'pos identical',
                  all(v for k, v in R.items() if k.startswith('pos|') and k.endswith('identical')), flush=True)


if __name__ == '__main__':
    main()
