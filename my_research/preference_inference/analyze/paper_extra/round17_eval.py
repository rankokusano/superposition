"""
Paper round 17 Sec.2 (docs/paper_instructions_round17.md; docs/paper_extra_log.md Sec.41; exploratory, outside the
pre-registration; procedure committed before computing). No training: the stage-I predictor and G are fixed.
Per checkpoint: G's Q2 = the saved stage-C Q2 (bit-identical to G's own output, round 16 Sec.40.5); cross-fitted
regressions of Q2 on the behaviour sets (round17_feat; free G: F0-F5, one-bit G: F0, F4, F5); then the round-16 loss
path (round16_eval.run: stage-B set, t = 0 vision shown, A-1 motion fed, eval mode, seed_all(0), cudnn
deterministic, batch 50; Q2 replaced at every step t = 0..99 through p2_override) with
  fit_F = clip(Q2_hat_F)   and   res_F = clip(Q2 - Q2_hat_F)   (clip to [-1, 1]; fractions saved).
Losses (self-vision L1, FP from h1, FP from h2) as episode means over steps 0..49 (main) and 0..99 (reference).
L(a) and L(z) are not recomputed: the round-16 saved values are used (identical path; checked in Sec.41.1).
Also saved per episode, steps 0..49: sum of squared residuals of Q2_hat_F and sum of squares of Q2 about the
checkpoint's per-dimension mean (for the bootstrap of R^2).
--models <exp>:<seed>:<epoch>[,<epoch>...] ...
-> data/result/paper_extra/Round17/eval/<exp>_ep<e>.npz (+ .json)
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
sys.path.insert(0, _HERE)
import round17_feat as F17  # noqa
R16 = F17.R16
C = R16.C
OUT = os.path.join(C.RES, 'paper_extra', 'Round17')


def evaluate(exp, s, e, dev, X, P2, P1):
    t0 = time.time()
    model, _ = C.load_v12(exp, s, e, dev)
    model.eval()
    onebit = type(model).__name__.endswith('OneBit')
    d = C.E.load_set('vdirb')
    g = h5py.File(d['path'], 'r')[d['split']]
    N = g['self_vision'].shape[0]
    Qa = np.load(os.path.join(C.OUT, 'eval', f'{exp}_ep{e}.npz'))['Q2']
    sets = ('F0', 'F4', 'F5') if onebit else F17.SETS
    fits, finfo = F17.fit_all(Qa, X, P2, P1, onebit, sets)
    t_fit = time.time() - t0
    info = {'exp': exp, 'seed': s, 'epoch': e, 'onebit': bool(onebit), 'N': N, 'sets': list(sets), 'fit': finfo,
            'clip': {}, 't_fit_s': t_fit}
    sav = {}
    y = Qa[:, :50].astype(np.float64)
    mu = y.reshape(-1, 8).mean(0)
    sav['ss_tot'] = ((y - mu) ** 2).sum((1, 2))
    for f in sets:
        sav[f'ss_res|{f}'] = ((y - fits[f][:, :50].astype(np.float64)) ** 2).sum((1, 2))
        for kind, Q in (('fit', fits[f]), ('res', Qa - fits[f])):
            q, info['clip'][f'{f}|{kind}'] = R16.clip(Q)
            L = R16.run(model, g, N, dev, q)
            for w, sl in R16.WIN.items():
                for i, nm in enumerate(('L1', 'FP1', 'FP2')):
                    sav[f'{f}|{kind}|{w}|{nm}'] = L[:, sl, i].mean(1).astype(np.float32)
    info['t_s'] = time.time() - t0
    return sav, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=[])
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    X, P2, P1, _ = F17.variables()
    os.makedirs(os.path.join(OUT, 'eval'), exist_ok=True)
    for spec in a.models:
        exp, s, eps = spec.split(':')
        for e in map(int, eps.split(',')):
            sav, info = evaluate(exp, int(s), e, dev, X, P2, P1)
            tag = f'{exp}_ep{e}'
            np.savez_compressed(os.path.join(OUT, 'eval', f'{tag}.npz'), **sav)
            json.dump(info, open(os.path.join(OUT, 'eval', f'{tag}.json'), 'w'), indent=1)
            print(tag, 'fit', round(info['t_fit_s'], 1), 't', round(info['t_s'], 1), flush=True)


if __name__ == '__main__':
    main()
