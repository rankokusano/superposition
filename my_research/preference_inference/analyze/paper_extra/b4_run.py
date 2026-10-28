"""
Paper extra B4 (docs/paper_extra_log.md Sec.29): the existing evaluation functions run on test data made like
A-1's own 80% stage-1 data, without changing them. Only the data paths they read are replaced:
  * eval_trial_v12.COMMON / eval_g5_v12.COMMON (common test v11_s1_random) -> v13_s1_mixed80 test (A-2 stationary);
  * eval_s2_v11.SETS green / pref (v11_s2t_green, v11_s2_pref test) -> paper_s2t_green_a1p80 / paper_s2_pref_a1p80
    (A-1 rule p 0.8 liked Red, A-2 as the originals; simulation/collect_data_paper_b4.py).
Models: mixed80 (v13_s1_mixed), signrand (paper_extra/C_signrand/s1_signrand), random (v12_s1_random); seed = pair.
The signed-pleasure class is registered as in c_run.py (evaluation data carry no a1_liked_t -> sign +1).
Outputs under data/result/paper_extra/B4/.
Usage (inside the container, from my_research/preference_inference; CUDA_VISIBLE_DEVICES = one GPU):
    python analyze/paper_extra/b4_run.py track --models mixed80:0 ... [--epochs ...] [--common] [--n N] [--tag T]
          h1 -> A-1 position tracking (eval_trial_v12.run + tracking) on the mse checkpoints; --common: on the
          original common test (check against the saved trial evaluation)
    python analyze/paper_extra/b4_run.py g5b   --models mixed80:0 ... [--epochs ...] [--n N] [--tag T]
          G5b only (eval_g5_v12.g5b) on the l1 checkpoints
    python analyze/paper_extra/b4_run.py g5    <eval_g5_v13.py arguments>   (G5a on the new sets + G5b again)
    python analyze/paper_extra/b4_run.py decomp --seed <s> --conds ...       (flipped G5a on the new sets)
--n N uses the first N test episodes (timing runs; written under B4/smoke/).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
for p in (_HERE, os.path.join(_PI, 'analyze', 'v13'), os.path.join(_PI, 'analyze', 'v12'), os.path.join(_PI, 'analyze', 'v11'), _PI):
    sys.path.insert(0, p)

from model import model_v12, model_v14  # noqa
if not hasattr(model_v12, 'SuperpositionNetworkMotionPleasureHoldSigned'):
    model_v12.SuperpositionNetworkMotionPleasureHoldSigned = model_v14.SuperpositionNetworkMotionPleasureHoldSigned
import eval_trial_v12  # noqa
import eval_g5_v12  # noqa
import eval_s2_v11  # noqa

RES = eval_trial_v12.RES
DATA = os.path.join(_PI, 'data', 'data')
OUT = os.path.join(RES, 'paper_extra', 'B4')
ORIG_COMMON = eval_trial_v12.COMMON
COMMON80 = os.path.join(DATA, 'v13_s1_mixed80', 'data.h5')
eval_trial_v12.COMMON = eval_g5_v12.COMMON = COMMON80
eval_s2_v11.SETS = dict(eval_s2_v11.SETS, green=('paper_s2t_green_a1p80', 'test'), pref=('paper_s2_pref_a1p80', 'test'))
L1 = {'mixed80': 'v13_s1_mixed_l1_s{s}', 'signrand': 'paper_extra/C_signrand/s1_signrand_l1_s{s}',
      'random': 'v12_s1_random_l1_s{s}'}
MSE = {'mixed80': 'v13_s1_mixed_mse', 'signrand': 'paper_extra/C_signrand/s1_signrand_mse', 'random': 'v12_s1_random_mse'}


class _Head(object):
    """h5py group view limited to the first n episodes (timing runs)."""

    def __init__(self, g, n):
        self.g, self.n = g, n

    def __getitem__(self, k):
        return _HeadDS(self.g[k], self.n)


class _HeadDS(object):
    def __init__(self, ds, n):
        self.ds, self.n = ds, n
        self.shape = (min(n, ds.shape[0]),) + ds.shape[1:]

    def __getitem__(self, idx):
        if idx == ():
            return self.ds[:self.n]
        if isinstance(idx, slice):
            a, b, _ = idx.indices(self.shape[0])
            return self.ds[a:b]
        return self.ds[:self.n][idx]


def _limit(n):
    """Make h5py.File(...)['test'] in the evaluation modules return the first n episodes."""
    import h5py
    real = h5py.File

    class F(object):
        def __init__(self, *a, **k):
            self.f = real(*a, **k)

        def __getitem__(self, k):
            return _Head(self.f[k], n)
    for mod in (eval_trial_v12, eval_g5_v12):
        mod.h5py = type('h5', (), {'File': F})


def common_args(ap):
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--epochs', type=int, nargs='+', default=[160, 170, 180, 190, 200])
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--tag', default='')


def track():
    ap = argparse.ArgumentParser()
    common_args(ap)
    ap.add_argument('--common', action='store_true')
    a = ap.parse_args()
    if a.common:
        eval_trial_v12.COMMON = ORIG_COMMON
    if a.n:
        _limit(a.n)
    out = os.path.join(OUT, 'smoke' if a.n else '', 'track_common' if a.common else 'track')
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    for spec in a.models:
        c, s = spec.split(':')
        for e in a.epochs:
            t0 = time.time()
            m, pm = eval_trial_v12.load_v12(MSE[c], int(s), e, dev)
            r = eval_trial_v12.tracking(eval_trial_v12.run(m, pm, dev))
            r.update(exp=MSE[c], seed=int(s), epoch=e, data=eval_trial_v12.COMMON, seconds=time.time() - t0)
            with open(os.path.join(out, f'{c}_s{s}_ep{e}{a.tag}.json'), 'w') as f:
                json.dump(r, f, indent=1)
            print(c, s, e, 'p_next visible', round(r['p_next']['visible'], 4), f'{r["seconds"]:.0f}s', flush=True)


def g5b():
    ap = argparse.ArgumentParser()
    common_args(ap)
    a = ap.parse_args()
    if a.n:
        _limit(a.n)
    out = os.path.join(OUT, 'smoke' if a.n else '', 'g5b')
    os.makedirs(out, exist_ok=True)
    dev = torch.device('cuda:0')
    torch.backends.cudnn.deterministic = True
    for spec in a.models:
        c, s = spec.split(':')
        for e in a.epochs:
            t0 = time.time()
            m, pm = eval_trial_v12.load_v12(L1[c].format(s=s), int(s), e, dev)
            m.motion_masking = True      # canonical for G5b, as eval_g5_v13
            r, main, main_bg = eval_g5_v12.g5b(m, pm, dev)
            r.update(exp=L1[c].format(s=s), seed=int(s), epoch=e, data=eval_g5_v12.COMMON, seconds=time.time() - t0)
            np.savez_compressed(os.path.join(out, f'{c}_s{s}_ep{e}{a.tag}.npz'), G5b=main, G5b_bg=main_bg)
            with open(os.path.join(out, f'{c}_s{s}_ep{e}{a.tag}.json'), 'w') as f:
                json.dump(r, f, indent=1)
            print(c, s, e, 'G5b', round(r['main_pm_own']['mean'], 4), 'frac_read_ctrl', round(r['frac_read_ctrl'], 3),
                  f'{r["seconds"]:.0f}s', flush=True)


def main():
    which = sys.argv.pop(1)
    sys.argv[0] = __file__
    if which == 'track':
        return track()
    if which == 'g5b':
        return g5b()
    if which == 'g5':
        import eval_g5_v13 as mod
    elif which == 'decomp':
        import g5a_decomp_v13 as mod
        mod.EXPS = dict(L1)
        mod.RES = os.path.join(OUT, 'decomp')        # -> .../decomp/v13/post/g5a_decomp/s<s>
    else:
        raise SystemExit(f'unknown {which}')
    sys.argv[0] = mod.__file__
    mod.main()


if __name__ == '__main__':
    main()
