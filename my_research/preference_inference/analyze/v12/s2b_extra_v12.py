"""
v12 stage 2b descriptive measures registered in docs/v12_experiment_log.md Sec.35 (bins 2-5 are 0 by construction):
  * sign agreement: over frames x bins 0,1,6,7 where A-2's true pleasure P_A2 is non-zero, the fraction where
    sign(Q2) == sign(P_A2), overall and per layout;
  * spread inside bins 0,1,6,7: per-bin std, between-episode / within-episode (time) std.
Generated Q2 on v11_s2t_green (test, 300 ep), G late-5, stage-2 protocol (as eval_s2_v11 via s2run).
Usage: python analyze/v12/s2b_extra_v12.py --prefix v12_s2b --seeds 0 2 --out stage2b
"""

import argparse
import json
import os
import statistics as st
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'v11'))
from eval_trial_v12 import RES, load_v12  # noqa
from eval_s2_v11 import load_set, true_pleasure  # noqa
from figs_s2extra_v12 import run  # noqa

BINS = [0, 1, 6, 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='v12_s2b')
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--out', default='stage2b')
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    d = load_set('green')
    P = true_pleasure('green', d)[:, 1:100][..., BINS]
    lay = np.repeat(d['layout'][:, None], 99, 1)
    res = {}
    for s in a.seeds:
        for stg in ('green', 'pref'):
            for c in ('random', 'mixed'):
                rows = []
                for e in a.epochs:
                    m, _ = load_v12(f'{a.prefix}_{stg}_{c}_s{s}', s, e, dev)
                    Q, _ = run(m, d, dev)
                    Q = Q[:, 1:100][..., BINS]
                    nz = P != 0
                    agree = np.sign(Q) == np.sign(P)
                    r = {'sign_agree_all': float(agree[nz].mean())}
                    for L, nm in ((0, 'original'), (1, 'swap_rg')):
                        mk = nz & (lay[..., None] == L)
                        r[f'sign_agree_{nm}'] = float(agree[mk].mean())
                    r['std_per_bin_0167'] = Q.reshape(-1, 4).std(0).tolist()
                    r['std_between_episode'] = float(Q.mean(1).std(0).mean())
                    r['std_within_episode_time'] = float(Q.std(1).mean())
                    r['frac_u_positive'] = float((Q[..., 2] > 0).mean())   # sign of bins 6,7 (= u)
                    rows.append(r)
                agg = {k: st.mean(x[k] for x in rows) for k in rows[0] if not isinstance(rows[0][k], list)}
                agg['std_per_bin_0167'] = np.mean([x['std_per_bin_0167'] for x in rows], 0).round(3).tolist()
                res[f'{stg}_{c}_s{s}'] = agg
                print(stg, c, s, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()}), flush=True)
    os.makedirs(os.path.join(RES, 'v12', a.out, 'summary'), exist_ok=True)
    json.dump(res, open(os.path.join(RES, 'v12', a.out, 'summary', 's2b_extra_' + ''.join(map(str, a.seeds)) + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
