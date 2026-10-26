"""
v15 exploratory checks after stage 2 (methods written before computing: docs/v15_experiment_log.md Sec.24.3, 24.4).
No training. For one tag (or its before-training reference with --ref0):
  2(a) P2-c with G's input set to 0 after eating (network vision masked, fed frame frozen at t_e) on the generated
       pairs and the control (3) pairs; memory readout (h2 and G's LSTM state) under that protocol.
  4    self side: P1-c only at eat moments inside a motion-hidden interval (pleasure held at its pre-eat value),
       next to the moments where the interval starts at t_e; other side: P2-a with process-2 = zeros and = candidate
       X's pleasure (for every episode), next to the true pleasure.

    python analyze/v15/eval_s2_extra_v15.py --tag exp_v4 [--ref0]     # -> data/result/v15/stage2/extra_<tag>.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from analyze.v15 import eval_s2_v15 as E  # noqa

K, T = E.K, E.T


def pairs_gzero(m2, dev, key, nx, ny):
    dx, dy, keep = E.pair_data(nx, ny)
    both, te, pair = E.stack_pairs(dx, dy, keep)
    N = len(te)
    pm = lambda t: np.full(N, 0. if t == 0 else 1.)  # noqa: E731
    gz = lambda t: t > te  # noqa: E731
    usual = E.run(m2, both['self_vision'], both['self_motion'], dev, pmask=pm)
    roll = E.run(m2, both['self_vision'], both['self_motion'], dev, pmask=pm, frame=lambda t: np.minimum(t, te), g_zero=gz)
    fo = [(pair % 2 == 0), (pair % 2 == 1)]
    p_hat = E.readout(usual['h2'], both['other_position'], np.ones((N, 100), bool), roll['h2'], fo)
    acc, blue, norms = [], [], []
    for i in range(N):
        t_e = te[i]
        dd = p_hat[i, t_e + K - 1] - p_hat[i, t_e - 1]
        lay, p0 = both['layout'][i], both['other_position'][i, t_e]
        cb = np.dot(E.unit(dd), E.unit(E.CUBE_C[lay][2] - p0)); cc = np.dot(E.unit(dd), E.unit(E.CUBE_C[lay][3] - p0))
        truth = both['a2_target'][i, t_e]
        acc.append(float((cb > cc) == (truth == 2)) if truth in (2, 3) else None)
        blue.append(float(cb > cc)); norms.append(float(np.linalg.norm(dd)))
    member = np.array(['first'] * (N // 2) + ['second'] * (N // 2))
    out = {'n_pairs': len(keep), 'pred_disp_norm_median': float(np.median(norms))}
    if key == 'gen':
        out['acc'] = E.split_boot(acc, pair, ['all'] * N)['all']
        out['acc_by_member'] = E.split_boot(acc, pair, member)
        out['toward_blue'] = E.split_boot(blue, pair, member)
        out['memory_readout'] = E.memory_readout(m2, dev, roll, both, te, pair)
    else:
        H = N // 2
        out['max_input_diff_upto_te'] = max(int(np.abs(both['self_vision'][i, :te[i] + 1].astype(int) - both['self_vision'][i + H, :te[i] + 1]).max()) for i in range(H))
        out['max_h2_diff'] = max(float(np.abs(roll['h2'][i, :te[i] + K] - roll['h2'][i + H, :te[i] + K]).max()) for i in range(H))
    return out


def self_side(m1, dev):
    d = E.read('v15_s1_p80_half_3000', 'test', ['self_vision', 'self_motion', 'self_position', 'a1_eat', 'a1_target', 'available', 'layout'])
    N = d['self_vision'].shape[0]
    fo = E.folds(N)
    hid = E.schedule(m1, N)
    usual = E.run(m1, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 0.99), hidden=lambda t: hid[:, t])
    ev = E.first_informative(d['a1_eat'], d['a1_target'])
    te = np.array([ev.get(n, (10 ** 6, -1))[0] for n in range(N)])
    roll = E.run(m1, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.where(t >= te, 1.0, 0. if t == 0 else 0.99),
                 hidden=lambda t: np.where(t >= te, True, hid[:, t]))
    p_hat = E.readout(usual['h1'], d['self_position'], np.ones((N, 100), bool), roll['h1'], fo)
    rows = {'held_pre_eat': ([], [], []), 'held_from_te': ([], [], [])}
    for n, (t_e, col) in ev.items():
        group = 'held_pre_eat' if hid[n, t_e - 1] else 'held_from_te'
        dd = p_hat[n, t_e + K - 1] - p_hat[n, t_e - 1]
        df, _ = E.bc_metrics(dd, d['self_position'][n, t_e], d['layout'][n], d['available'][n, t_e], d['a1_target'][n, t_e])
        v, ids, cells = rows[group]
        v.append(df); ids.append(n); cells.append(E.eaten_corner(d['layout'][n], col))
    return {g: E.cell_boot(v, ids, cells) for g, (v, ids, cells) in rows.items()}


def other_side(m1, dev):
    keys = ['self_vision', 'other_vision', 'self_motion', 'other_position', 'a2_eat', 'a2_target', 'available', 'layout', 'a2_pleasure', 'a2_cand']
    d = E.read('v15_s2_c2_p80_half_6000', 'test', keys)
    N = d['self_vision'].shape[0]
    fo = E.folds(N)
    m1.motion_masking = False
    ev = E.first_informative(d['a2_eat'], d['a2_target'])
    te = np.array([ev.get(n, (10 ** 6, -1))[0] for n in range(N)])
    ones, zeros2 = np.ones((N, 1)), np.zeros((N, 2))
    pX = E.cand_pleasure(d['other_vision'])['X']
    sources = {'true': d['a2_pleasure'], 'zero': np.zeros_like(d['a2_pleasure']), 'X_for_all': pX}
    out = {}
    for name, P in sources.items():
        usual = E.run(m1, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.zeros(N),
                      p2=lambda t, P=P: np.concatenate([zeros2, P[:, t], ones], 1))
        held = lambda t, P=P: np.where((t >= te)[:, None], P[np.arange(N), np.minimum(te, T - 1)], P[:, t])  # noqa: E731
        roll = E.run(m1, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.where(t >= te, 1.0, 0.),
                     p2=lambda t, held=held: np.concatenate([zeros2, held(t), ones], 1))
        p_hat = E.readout(usual['h2'], d['other_position'], np.ones((N, 100), bool), roll['h2'], fo)
        diffs, blue, ids, grp, cells = [], [], [], [], []
        for n, (t_e, col) in ev.items():
            dd = p_hat[n, t_e + K - 1] - p_hat[n, t_e - 1]
            df, bl = E.bc_metrics(dd, d['other_position'][n, t_e], d['layout'][n], d['available'][n, t_e], d['a2_target'][n, t_e])
            g = 'X' if d['a2_cand'][n] == 0 else 'Y'
            diffs.append(df); blue.append(bl); ids.append(n); grp.append(g); cells.append(E.eaten_corner(d['layout'][n], col) + '_' + g)
        out[name] = {'toward_blue': E.split_boot(blue, ids, grp), 'diff_by_corner_cand': E.cell_boot(diffs, ids, cells)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ref0', action='store_true')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    torch.set_num_threads(3)
    dev = torch.device(a.device)
    if a.ref0:
        get1 = lambda: E.load_ref0(a.tag, dev)[0]  # noqa: E731
        m2 = E.load_ref0(a.tag, dev)[1]
        name = f'{a.tag}_ref0'
    else:
        get1 = lambda: E.load(f'v15_s2p1_{a.tag}_mse', 150, dev)  # noqa: E731
        m2 = E.load(f'v15_s2p2_{a.tag}', 150, dev)
        name = a.tag
    R = {'tag': name, 'device': a.device}
    R['2a_gen'] = pairs_gzero(m2, dev, 'gen', 'v15_s2_genx_p80', 'v15_s2_geny_p80'); print('2a gen', R['2a_gen']['acc'], flush=True)
    R['2a_ctrl3'] = pairs_gzero(m2, dev, 'ctrl3', 'v15_s2_c3x_p80', 'v15_s2_c3xp_p80'); print('2a ctrl3', R['2a_ctrl3'], flush=True)
    R['4_self'] = self_side(get1(), dev); print('4 self', {g: v['equal_weight'] for g, v in R['4_self'].items()}, flush=True)
    R['4_other'] = other_side(get1(), dev); print('4 other done', flush=True)
    json.dump(R, open(os.path.join(E.OUT, f'extra_{name}.json'), 'w'), indent=1)
    print('saved', name, flush=True)


if __name__ == '__main__':
    main()
