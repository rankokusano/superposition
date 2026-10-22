"""
v13 descriptive self-position tracking (docs/v13_experiment_log.md Sec.3.3 end, Sec.7); not a gate.
G1's measure (analyze/v12/eval_trial_v12.py: run + h1 -> p_{t+1}, Ridge(1), episode GroupKFold(5) over
all frames of the test set) on three test sets, each 300 episodes:
    common  = v11_s1_random test (the gate's test), mixed50 = v11_s1_mixed test, mixed80 = v13_s1_mixed80 test,
and the cross-validated estimates scored on subsets of frames:
    visible / hidden  x  near red (|p_{t+1} - red reference| < 5, the episode's layout) / away.
Reports R^2 within each subset and the median position error (R^2 within a subset is lower when the
positions are less spread, so the error is shown as well) and the subset's frame count.
Writes data/result/v13/<out>/track_<cond>_ep<e>.json.
Run through analyze/v13/v12run.py track (maps v13 model names).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v12'))
sys.path.insert(0, os.path.join(_PI, 'analyze', 'v11'))
import eval_trial_v12 as etv  # noqa
from eval_trial_v12 import DATA, RES, load_v12  # noqa
import gate61_v11 as g11  # noqa

TESTS = {'common': 'v11_s1_random', 'mixed50': 'v11_s1_mixed', 'mixed80': 'v13_s1_mixed80'}


def scores(Y, Yh, m):
    if m.sum() < 10:
        return {'n': int(m.sum())}
    return {'r2': float(1 - ((Y[m] - Yh[m]) ** 2).sum() / ((Y[m] - Y[m].mean(0)) ** 2).sum()),
            'err_median': float(np.median(np.linalg.norm(Y[m] - Yh[m], axis=1))), 'n': int(m.sum())}


def split_tracking(d):
    N = d['h'].shape[0]
    X = d['h'].reshape(-1, 128)
    grp = np.repeat(np.arange(N)[:, None], 100, 1).reshape(-1)
    Y = d['pos'][:, 1:101].reshape(-1, 2)
    Yh = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y, grp):
        Yh[te] = Ridge(alpha=1.0).fit(X[tr], Y[tr]).predict(X[te])
    red = np.stack([g11.RED_REF[int(l)] for l in d['layout']])[:, None, :]
    near = (np.linalg.norm(d['pos'][:, 1:101] - red, axis=-1) < 5).reshape(-1)
    hid = d['hidden'].reshape(-1)
    out = {'near_frac': float(near.mean())}
    for hn, hm in (('all', np.ones_like(hid)), ('visible', ~hid), ('hidden', hid)):
        out[hn] = {'all': scores(Y, Yh, hm), 'near_red': scores(Y, Yh, hm & near), 'away': scores(Y, Yh, hm & ~near)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_fmt', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--conds', nargs='+', default=['random', 'mixed'])
    ap.add_argument('--tests', nargs='+', default=list(TESTS))
    ap.add_argument('--out', required=True, help='dir under data/result (e.g. v13/main/track/s0)')
    args = ap.parse_args()
    out_dir = os.path.join(RES, args.out)
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    for e in args.epochs:
        for c in args.conds:
            model, p_mask = load_v12(args.exp_fmt.format(c), args.seed, e, dev)
            r = {'exp': args.exp_fmt.format(c), 'seed': args.seed, 'epoch': e}
            for tn in args.tests:
                etv.COMMON = os.path.join(DATA, TESTS[tn], 'data.h5')   # run() reads this test set
                d = etv.run(model, p_mask, dev)
                r[tn] = split_tracking(d)
                del d
            with open(os.path.join(out_dir, f'track_{c}_ep{e}.json'), 'w') as f:
                json.dump(r, f, indent=1)
            print(c, e, json.dumps({tn: {k: r[tn]['visible'][k].get('r2') for k in ('all', 'near_red', 'away')}
                                    for tn in args.tests}), flush=True)


if __name__ == '__main__':
    main()
