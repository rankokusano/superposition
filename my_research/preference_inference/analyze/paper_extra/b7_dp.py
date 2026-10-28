"""
Paper round 5, work 1 (B7; docs/paper_extra_log.md Sec.31): d_P = (g_hold - g_flip) / 2 of the preference-switch
condition (v14-C) and of the 50% condition, without changing existing code.
  pref       v14_s1_mixedpref_mse   signed class, liked colour per episode (LikedFeed), g relative to the liked colour
  nonesigned v12_s1_random_mse      the same signed input and g as pref (v14 log Sec.31, main pairing)
  noexpv     v14_s1_randompref_mse  (pairs 1-2, descriptive) as pref
  mixed50    v12_s1_mixed_mse       plain class, sign +1, g relative to Red (as c_run.g3flip)
G3 input freeze K = 15 (t < 20 canonical) with the pleasure held (hold), zero (zero) or held with the sign flipped
(flip): c_run.freeze_flip; d* / slopes as localise_v11; g by gates_v14c.by_liked (signed) or localise_v11.g_of (50%).
Common test = v14_s1_randompref test (v11_s1_random test + a1_liked: 169 Red, 131 Green), as gates_v14c.
-> data/result/paper_extra/B7/<cond>_s<s>_ep<e><tag>.npz (hold, zero, flip, dP, liked)
Usage (inside the container, one GPU): python analyze/paper_extra/b7_dp.py --models pref:0 nonesigned:0 ... [--epochs ...] [--tag _rep]
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v14'))
sys.path.insert(0, _HERE)
import gates_v14c as GV  # noqa  (sets eval_trial_v12.COMMON to the v14_s1_randompref test)
import c_run  # noqa
from localise_v11 import dstar, slopes, g_of  # noqa

RES = os.path.join(_PI, 'data', 'result')
OUT = os.path.join(RES, 'paper_extra', 'B7')
EXP = {'pref': 'v14_s1_mixedpref_mse', 'nonesigned': 'v12_s1_random_mse', 'noexpv': 'v14_s1_randompref_mse',
       'mixed50': 'v12_s1_mixed_mse'}
K = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    fv = h5py.File(os.path.join(RES, 'v11', 'cf', 'cf_dist_views.h5'), 'r')
    views, vvalid = fv['views'][()], fv['valid'][()]
    liked = h5py.File(GV.COMMON, 'r')['test']['a1_liked'][()].astype(int)
    L = np.where(liked == 1, 0, 1)                  # liked colour index in COLORS (Red 0, Green 1)
    for spec in a.models:
        c, s = spec.split(':')
        for e in a.epochs:
            t0 = time.time()
            per = {'liked': liked}
            if c == 'mixed50':
                m, pm = GV.T.load_v12(EXP[c], int(s), e, dev)
                for mode in ('hold', 'zero', 'flip'):
                    per[mode] = g_of(slopes(dstar(c_run.freeze_flip(m, pm, dev, K, mode), views, vvalid), K))
            else:
                model, pm = GV.load_signed(EXP[c], int(s), e, dev)
                m = GV.LikedFeed(model)
                for mode in ('hold', 'zero', 'flip'):
                    m.set_episodes(liked)
                    sl = slopes(dstar(c_run.freeze_flip(m, pm, dev, K, mode), views, vvalid), K)
                    per[mode], _ = GV.by_liked({X: sl[:, i] for i, X in enumerate(GV.COLORS)}, L, True)
            per['dP'] = (per['hold'] - per['flip']) / 2
            np.savez_compressed(os.path.join(OUT, f'{c}_s{s}_ep{e}{a.tag}.npz'), **per)
            print(c, s, e, {q: round(float(np.nanmean(per[q])), 4) for q in ('hold', 'zero', 'flip', 'dP')},
                  f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
