"""
v15 2(b) evaluation (method: docs/v15_experiment_log.md Sec.24.5). Phase-2 models on the generated pairs and the control (3)
pairs with A-2 hidden in the INPUT view from t_e-5 to the end (data/data/<set>/hideA2_test.h5). Training protocol
(vision masked for t > 0; G sees the encoder output of the input view), true (A-2-hidden) frames throughout. Pairs whose
inputs differ between t_e and t_e + K (e.g. a cube eaten within the window) are dropped and counted.
Measures as in P2-c: pair accuracy (blue vs cyan from the h2 displacement over K = 10 steps), by member, toward blue,
readout of the candidate from h2 / G's LSTM state at t_e + {0,1,3,5,10} (readout fitted on the 2(b) training data,
first informative moment of each episode, input view with A-2 hidden); control (3): input / prediction identity.

    python analyze/v15/eval_s2_hide_v15.py --exp v15_s2p2h_exp_v4 [--epoch 150]        # trained 2(b) model
    python analyze/v15/eval_s2_hide_v15.py --exp v15_s2p2_exp_v4                       # phase-2 model trained without hiding
    python analyze/v15/eval_s2_hide_v15.py --ref0 exp_v4                               # before training
Writes data/result/v15/stage2/hide_<name>.json.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15 import eval_s2_v15 as E  # noqa

K, T = E.K, E.T
OFFSETS = (0, 1, 3, 5, 10)


def hidden_pairs(nx, ny):
    dx, dy, keep = E.pair_data(nx, ny)
    for d, name in ((dx, nx), (dy, ny)):
        with h5py.File(os.path.join(E.DATA, name, 'hideA2_test.h5'), 'r') as f:
            d['self_vision_in'] = f['train']['self_vision_in'][()]
    keep2, dropped = [], 0
    for n, t_e in keep:
        # Sec.24.5: the two inputs must be identical from t_e to t_e + K (before eating they differ in the top-left colour)
        if np.array_equal(dx['self_vision_in'][n, t_e:t_e + K + 1], dy['self_vision_in'][n, t_e:t_e + K + 1]) and \
                np.array_equal(dx['self_motion'][n, :t_e + K + 1], dy['self_motion'][n, :t_e + K + 1]):
            keep2.append((n, t_e))
        else:
            dropped += 1
    both, te, pair = E.stack_pairs(dx, dy, keep2)
    return both, te, pair, dropped


def readout_cand(m2, dev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    with h5py.File(os.path.join(E.DATA, 'v15_s2_c2_p80_half_1000t_hideA2', 'data.h5'), 'r') as f:
        g = f['train']
        d = {k: g[k][()] for k in ('self_vision_in', 'self_motion', 'a2_eat', 'a2_target', 'a2_cand')}
    N = d['self_vision_in'].shape[0]
    usual = E.run(m2, d['self_vision_in'], d['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 1.))
    ev = E.first_informative(d['a2_eat'], d['a2_target'])
    clfs = {}
    for key in ('h2', 'g'):
        X = np.concatenate([usual[key][n, min(t + o, T - 2)][None] for n, (t, _) in ev.items() for o in OFFSETS])
        y = np.array([d['a2_cand'][n] for n, _ in ev.items() for o in OFFSETS])
        clfs[key] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X, y)
    return clfs


def evaluate(m2, dev):
    R = {}
    clfs = readout_cand(m2, dev)
    for key, (nx, ny) in (('gen', ('v15_s2_genx_p80', 'v15_s2_geny_p80')), ('ctrl3', ('v15_s2_c3x_p80', 'v15_s2_c3xp_p80'))):
        both, te, pair, dropped = hidden_pairs(nx, ny)
        N = len(te); H = N // 2
        pm = lambda t: np.full(N, 0. if t == 0 else 1.)  # noqa: E731
        roll = E.run(m2, both['self_vision_in'], both['self_motion'], dev, pmask=pm)
        fo = [(pair % 2 == 0), (pair % 2 == 1)]
        p_hat = E.readout(roll['h2'], both['other_position'], np.ones((N, 100), bool), roll['h2'], fo)
        out = {'n_pairs': H, 'dropped_pairs_input_differs_before_te_plus_K': dropped}
        if key == 'ctrl3':
            out['max_h2_diff_upto_te_plus_K'] = max(float(np.abs(roll['h2'][i, :te[i] + K] - roll['h2'][i + H, :te[i] + K]).max()) for i in range(H))
            R[key] = out
            continue
        acc, blue, norms = [], [], []
        for i in range(N):
            t_e = te[i]
            dd = p_hat[i, t_e + K - 1] - p_hat[i, t_e - 1]
            lay, p0 = both['layout'][i], both['other_position'][i, t_e]
            cb = np.dot(E.unit(dd), E.unit(E.CUBE_C[lay][2] - p0)); cc = np.dot(E.unit(dd), E.unit(E.CUBE_C[lay][3] - p0))
            truth = both['a2_target'][i, t_e]
            acc.append(float((cb > cc) == (truth == 2)) if truth in (2, 3) else None)
            blue.append(float(cb > cc)); norms.append(float(np.linalg.norm(dd)))
        member = np.array(['first'] * H + ['second'] * H)
        out.update({'acc': E.split_boot(acc, pair, ['all'] * N)['all'], 'acc_by_member': E.split_boot(acc, pair, member),
                    'toward_blue': E.split_boot(blue, pair, member), 'pred_disp_norm_median': float(np.median(norms)),
                    'max_h2_diff_upto_te': max(float(np.abs(roll['h2'][i, :te[i]] - roll['h2'][i + H, :te[i]]).max()) for i in range(H))})
        mem = {}
        for k2 in ('h2', 'g'):
            mem[k2] = {}
            for o in OFFSETS:
                Z = roll[k2][np.arange(N), np.minimum(te + o, T - 2)]
                mem[k2][str(o)] = E.boot_mean((clfs[k2].predict(Z) == both['a2_cand']).astype(float), pair)
        out['memory_readout'] = mem
        R[key] = out
        R['_acc_ev'] = np.array([np.nan if x is None else x for x in acc]).tolist()
        R['_pair'] = pair.tolist()
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default=None)
    ap.add_argument('--epoch', type=int, default=150)
    ap.add_argument('--ref0', default=None, help='tag (e.g. exp_v4): before-training reference')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    torch.set_num_threads(3)
    dev = torch.device(a.device)
    if a.ref0:
        m2 = E.load_ref0(a.ref0, dev)[1]
        name = f'{a.ref0}_ref0'
    else:
        m2 = E.load(a.exp, a.epoch, dev)
        name = a.exp
    R = {'name': name, 'device': a.device, **evaluate(m2, dev)}
    json.dump(R, open(os.path.join(E.OUT, f'hide_{name}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in R.items() if not k.startswith('_')}), flush=True)


if __name__ == '__main__':
    main()
