"""
v15 Sec.33.8 (descriptive, pre-registered before the (D) results): how much phase 1 makes the superposition LSTM weights
differ between experienced and no-experience, at phase-2 ep0 (= after phase 1). Relative difference
||W_exp - W_noexp|| / ||W_noexp|| (Frobenius) per block of the LSTMCell: input columns of the vision (64), input columns of
the 11-dim motion / pleasure / flag input, recurrent weights, biases (bias_ih + bias_hh), and the last three together.
Shared models: superposition_module.lstm. NS models: lstm_other (the other's side) and lstm (the self side, reference).
Also checks that the phase-2 ep0 LSTM equals the phase-1 ep150 LSTM. CPU only.

    python analyze/v15/weights_ns_v15.py [--ns]   # --ns: include the NS runs -> data/result/v15/stage2/weights_lstm_v15i.json
"""

import argparse
import json
import os

import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(_PI_DIR, 'data', 'result')
S2 = os.path.join(RES, 'v15', 'stage2')
NV = 64  # vision columns first (SuperpositionModule: cat([vision, motion]))


def ck(exp, seed, epoch):
    return torch.load(os.path.join(RES, exp, str(seed), 'model', f'{epoch:05d}.pth'), map_location='cpu')['model']


def blocks(sd, mod):
    p = f'superposition_module.{mod}.'
    wi = sd[p + 'weight_ih']
    assert wi.shape[1] == NV + 11
    return {'vision_cols': wi[:, :NV], 'in11_cols': wi[:, NV:], 'recurrent': sd[p + 'weight_hh'],
            'bias': torch.cat([sd[p + 'bias_ih'], sd[p + 'bias_hh']])}


def rel(a, b):
    return float((a - b).norm() / b.norm())


def compare(e, n, mod):
    be, bn = blocks(e, mod), blocks(n, mod)
    out = {k: rel(be[k], bn[k]) for k in be}
    rest = ('vision_cols', 'recurrent', 'bias')
    out['rest'] = rel(torch.cat([be[k].flatten() for k in rest]), torch.cat([bn[k].flatten() for k in rest]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ns', action='store_true')
    a = ap.parse_args()
    R = {}
    kinds = [('shared', '', ['lstm'])] + ([('ns', '_ns', ['lstm_other', 'lstm'])] if a.ns else [])
    for kind, tag, mods in kinds:
        for s in (0, 1, 2):
            suf = tag + ('' if s == 0 else f'_s{s}')
            sd = {c: ck(f'v15_s2p2h_{c}_v15i{suf}', s, 0) for c in ('exp', 'noexp')}
            p1 = {c: ck(f'v15_s2p1_{c}_v15i{tag}_mse', s, 150) for c in ('exp', 'noexp')}
            same = all(torch.equal(sd[c][k], p1[c][k]) for c in sd for k in p1[c] if k.startswith('superposition_module.'))
            for mod in mods:
                R[f'{kind}_{mod}_seed{s}'] = {**compare(sd['exp'], sd['noexp'], mod), 'p2ep0_equals_p1ep150': same}
    json.dump(R, open(os.path.join(S2, 'weights_lstm_v15i.json'), 'w'), indent=1)
    cols = ('in11_cols', 'vision_cols', 'recurrent', 'bias', 'rest')
    print('model'.ljust(24), ' '.join(c.rjust(11) for c in cols), ' p2ep0==p1ep150')
    for k, v in R.items():
        print(k.ljust(24), ' '.join(f'{v[c]:11.4f}' for c in cols), ' ', v['p2ep0_equals_p1ep150'])


if __name__ == '__main__':
    main()
