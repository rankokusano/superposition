"""
v14 Sec.35 C (not pre-registered, no training; method recorded before computing): with A-2's true pleasure in
process-2 (G5a procedure: pref test, P_A2 frozen at t = 20, K = 15, A-2 read from the predicted image), does the
predicted A-2 move toward A-2's liked colour or toward A-1's own liked colour of the episode?
A-1's liked colour is held at Red and at Green (gates_v14c.LikedFeed); from the per-colour slopes s_X
(eval_g5_v12.g5a_metric) per episode: g_A2 = s_{A-2 liked} - mean(other 3), g_A1 = s_{A-1 liked} - mean(other 3),
split into episodes where A-1 and A-2 like the same colour / different colours (main). 1000 episode bootstrap CIs.
Models: v14-C experienced (s0-s2), v14-C no-experience version (s0, s1), v12 experienced (A-1 Red only: its input
is Red - Green). Writes data/result/v14/gates/g5a_whose.json.
"""

import json
import os
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import gates_v14c as GV  # noqa  (sets the v12 module paths)
G = GV.G
from p2test_v11 import calibrate  # noqa
from eval_trial_v12 import boot_ci  # noqa

COL = ('Red', 'Green', 'Blue', 'Cyan')


def g_of_colour(S, idx):
    n = np.arange(len(idx))
    others = np.stack([np.delete(S[i], idx[i]) for i in n])
    return S[n, idx] - np.nanmean(others, 1)


def summ(v):
    v = v[np.isfinite(v)]
    return {'mean': float(v.mean()), 'ci95': boot_ci(v), 'n': int(len(v))}


def main():
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    d = G.load_set('pref')
    P = G.true_pleasure('pref', d)
    gg = G.load_set('green')
    g = h5py.File(gg['path'], 'r')['test']
    ab = calibrate(g['self_vision'][:, :100:5], gg['self_position'][:, :100:5], gg['other_position'][:, :100:5])
    a2 = d['a2_liked'].astype(int)                      # 1 Red, 0 Green
    a2_idx = np.where(a2 == 1, 0, 1)
    models = [(f'v14c_exp_s{s}', f'v14_s1_mixedpref_l1_s{s}', s, (1, 0)) for s in (0, 1, 2)] + \
             [(f'v14c_noexpv_s{s}', f'v14_s1_randompref_l1_s{s}', s, (1, 0)) for s in (0, 1)] + \
             [(f'v12_exp_s{s}', f'v12_s1_mixed_l1_s{s}', s, (1,)) for s in (0, 1, 2)]
    out = {}
    for name, exp, s, a1s in models:
        for e in (160, 170, 180, 190, 200):
            m, _ = GV.load_signed(exp, s, e, dev)
            model = GV.LikedFeed(m)
            model.motion_masking = False
            for a1 in a1s:
                model.set_const(a1)
                est = G.g5a_est(model, d, P, dev, ab[0], ab[1], 15, 'freeze')
                _, _, S = G.g5a_metric(est, d)
                a1_idx = np.full(len(a2), 0 if a1 == 1 else 1)
                gA2, gA1 = g_of_colour(S, a2_idx), g_of_colour(S, a1_idx)
                same = (a2 == a1)
                r = {}
                for nm, mk in (('different', ~same), ('same', same), ('all', np.ones_like(same))):
                    r[nm] = {'g_A2': summ(gA2[mk]), 'g_A1': summ(gA1[mk])}
                key = f'{name}|a1{"red" if a1 else "green"}|ep{e}'
                out[key] = r
                print(key, 'different: g_A2 %+.3f %s  g_A1 %+.3f %s' % (r['different']['g_A2']['mean'], np.round(r['different']['g_A2']['ci95'], 3).tolist(),
                                                                      r['different']['g_A1']['mean'], np.round(r['different']['g_A1']['ci95'], 3).tolist()), flush=True)
    json.dump(out, open(os.path.join(GV.OUT, 'g5a_whose.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
