"""
v10 Fig.5 metrics: the same four errors as analyze/analyze_vpt.py
(`result.txt`: self_true_and_self_rec, other_true_and_other_rec,
self_true_and_other_rec, other_true_and_self_rec; mean over w,h,c per frame,
NaN frames dropped, then mean/std over frames; --margin crops the 21x21x21x21
grid on all four axes), computed in chunks and accepting uint8 grid data
(v10_swap_grid) as well as the float32 root grid.

  self_self   = |self_vision_no_agent  - self_rec|
  other_other = |other_vision_no_agent - other_rec|   (A-2 view recovered from A-1's view)
  self_other  = |self_vision_no_agent  - other_rec|
  other_self  = |other_vision_no_agent - self_rec|
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import git_hash  # noqa

PAIRS = {
    'self_self': ('self_vision_no_agent', 'self'),
    'other_other': ('other_vision_no_agent', 'other'),
    'self_other': ('self_vision_no_agent', 'other'),
    'other_self': ('other_vision_no_agent', 'self'),
}


def as_float(a):
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_h5', required=True)
    ap.add_argument('--saved_h5', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--mode', default='eval')
    ap.add_argument('--margin', type=int, default=1)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    fd = h5py.File(args.data_h5, 'r')['train']
    fs = h5py.File(args.saved_h5, 'r')[f'{args.epoch:05d}/{args.mode}']
    rec = {'self': fs['self_vision/reconstruction'], 'other': fs['other_vision/reconstruction']}
    n_out, n_in = fd['self_vision'].shape[:2]
    size = int(round(np.sqrt(n_out)))
    assert size * size == n_out == n_in
    assert rec['self'].shape[:2] == (n_out, n_in), rec['self'].shape
    m = args.margin
    keep_1d = np.arange(m, size - m)
    keep_in = (keep_1d[:, None] * size + keep_1d[None, :]).reshape(-1)  # inner (self) grid
    errs = {k: [] for k in PAIRS}
    for oi in keep_1d:
        for oj in keep_1d:
            n = oi * size + oj
            truth = {k: as_float(fd[k][n][keep_in]) for k in ('self_vision_no_agent', 'other_vision_no_agent')}
            r = {k: as_float(rec[k][n][keep_in]) for k in rec}
            for name, (tk, rk) in PAIRS.items():
                e = np.abs(truth[tk] - r[rk]).mean(axis=(1, 2, 3))
                errs[name].append(e[~np.isnan(e)])
    out = {'label': args.label, 'epoch': args.epoch, 'margin': m, 'mode': args.mode,
           'data_h5': args.data_h5, 'saved_h5': args.saved_h5, 'git_commit': git_hash()}
    for name, lst in errs.items():
        v = np.concatenate(lst)
        out[name] = {'mean': float(v.mean()), 'std': float(v.std()), 'n': int(v.size)}
    out['margin_other_self_minus_other_other'] = out['other_self']['mean'] - out['other_other']['mean']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
