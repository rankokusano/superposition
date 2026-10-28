"""
v15 Sec.36 (exploratory E1-E3): run one saved phase-2 model on CPU (no training) and save what E1-E3 need.
    - h1 on the label data (Sec.36.1): after the first eat until the next eat, frames where A-1 has a target; with the
      target corner and the episode id. Main: no-experience test (goalcap); secondary: rank X + rank Y tests.
    - on the A-2-hidden generated pairs (930 pairs): h2 and G's output q2 at t_e .. t_e + 10, A-2's view pleasure with the
      X / Y values, pair / member (P with the true values equals the stored a2_pleasure: checked, max diff 0.0, log Sec.37).
Same flow as the hidden test (eval_s2_v15.run, view only at t = 0, A-1's true motion).

    python analyze/v15/extract_e123_v15.py --exp v15_s2p2h_exp_v15i --seed 1   # -> data/result/v15/e123/<name>.npz
    python analyze/v15/extract_e123_v15.py --ref0                               # before-training model
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from model import model_v15, model_v15ns  # noqa: E402
for _name in ('SuperpositionNetworkMotionPleasureHoldV15NS', 'SuperpositionNetworkMotionPleasureHoldOtherV15NS'):
    setattr(model_v15, _name, getattr(model_v15ns, _name))
from analyze.v15 import eval_s2_hide_v15 as H  # noqa: E402
from analyze.v15 import eval_s2_v15 as E  # noqa: E402
from analyze.v15.stage0_v15 import _COL_CORNER  # noqa: E402

OUT = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'e123')
CORNERS = ('TL', 'BL', 'BR', 'TR')
LABEL_SETS = {'goalcap': [('v15_s1_goalcap_half_3000', 'test')],
              'rankXY': [('v15_s1_p80_half_3000', 'test'), ('v15_s1_rankY_p80_half_1000t', 'test')]}
OFFS = np.arange(0, 11)


def label_frames(eat, tgt, layout):
    """-> list of (episode, t, corner index) after the first eat until the next eat, A-1's target >= 0."""
    out = []
    for n in range(eat.shape[0]):
        ts = np.where(eat[n] >= 0)[0]
        if len(ts) == 0:
            continue
        t0 = ts[0] + 1
        t1 = ts[1] + 1 if len(ts) > 1 else E.T - 1
        for t in range(t0, min(t1, E.T - 1)):
            if tgt[n, t] >= 0:
                out.append((n, t, CORNERS.index(_COL_CORNER[int(layout[n])][int(tgt[n, t])])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ref0', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = torch.device('cpu')
    if a.ref0:
        m2, name = E.load_ref0('exp_v15i', dev)[1], 'ref0_v15i'
    else:
        m2 = E.load(a.exp, 150, dev, seed=a.seed)
        name = a.exp if a.seed == 0 else f'{a.exp}_seed{a.seed}'
    os.makedirs(OUT, exist_ok=True)
    R = {}
    for tag, sets in LABEL_SETS.items():
        H1, LAB, EP = [], [], []
        for k, (ds, split) in enumerate(sets):
            with h5py.File(os.path.join(E.DATA, ds, 'data.h5'), 'r') as f:
                g = f[split]
                d = {key: g[key][()] for key in ('self_vision', 'self_motion', 'a1_eat', 'a1_target', 'layout')}
            fr = label_frames(d['a1_eat'], d['a1_target'], d['layout'])
            N = d['self_vision'].shape[0]
            roll = E.run(m2, d['self_vision'], d['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 1.))
            H1.append(np.stack([roll['h1'][n, t] for n, t, _ in fr]))
            LAB.append(np.array([c for _, _, c in fr])); EP.append(np.array([k * 10000 + n for n, _, _ in fr]))
        R[f'{tag}_h1'] = np.concatenate(H1).astype(np.float32)
        R[f'{tag}_corner'] = np.concatenate(LAB).astype(np.int8)
        R[f'{tag}_episode'] = np.concatenate(EP).astype(np.int32)
    both, te, pair, dropped = H.hidden_pairs('v15_s2_genx_p80', 'v15_s2_geny_p80')
    N = len(te)
    roll = E.run(m2, both['self_vision_in'], both['self_motion'], dev, pmask=lambda t: np.full(N, 0. if t == 0 else 1.))
    idx = np.minimum(te[:, None] + OFFS[None], E.T - 2)
    rows = np.arange(N)[:, None]
    R['h2'] = roll['h2'][rows, idx].astype(np.float32)          # (N, 11, 128), offsets 0..10
    R['q2'] = roll['q2'][rows, idx].astype(np.float32)          # (N, 11, 8)
    ov = both['other_vision'][rows, idx]                        # (N, 11, 16, 64, 3)
    cp = E.cand_pleasure(ov)
    R['p_X'], R['p_Y'] = cp['X'].astype(np.float32), cp['Y'].astype(np.float32)
    R['pair'] = pair; R['member'] = np.array([0] * (N // 2) + [1] * (N // 2), np.int8); R['te'] = te
    np.savez_compressed(os.path.join(OUT, f'{name}.npz'), **R)
    print(name, {k: v.shape for k, v in R.items()}, 'dropped', dropped, flush=True)


if __name__ == '__main__':
    main()
