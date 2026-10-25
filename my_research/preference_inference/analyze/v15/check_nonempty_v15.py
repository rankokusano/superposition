"""
v15 data non-emptiness check (docs/v15_experiment_log.md Sec.20; after the empty no-experience data, Sec.19.0).
For every v15 dataset under data/data (and, with --all, the check files under data/result/v15/datacheck), every
split and EVERY episode: non-zero pixel fraction of self/other vision at the first and last frame, whether the
A-1 position moves or A-2's position is set (non-zero), and the non-zero fraction of the stored pleasure.
An episode is 'empty' if its first-frame self view is all zero. Prints one line per dataset/split; exit 1 if any
episode is empty. Run after every data generation from now on.

    python analyze/v15/check_nonempty_v15.py [--all] [names ...]
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(path):
    lines, bad = [], 0
    with h5py.File(path, 'r') as f:
        for sp in f:
            g = f[sp]
            n = g['self_vision'].shape[0]
            sv0 = g['self_vision'][:, 0]
            svl = g['self_vision'][:, -1]
            ov0 = g['other_vision'][:, 0]
            empty = ~sv0.reshape(n, -1).any(1)
            pos1 = g['self_position'][()]
            pos2 = g['other_position'][()]
            p1 = g['a1_pleasure'][()] if 'a1_pleasure' in g else None
            r = {'episodes': n, 'empty_eps': int(empty.sum()),
                 'sv_nonzero_first': float((sv0 > 0).mean()), 'sv_nonzero_last': float((svl > 0).mean()),
                 'ov_nonzero_first': float((ov0 > 0).mean()),
                 'a1_moves_frac': float((np.abs(np.diff(pos1, axis=1)).sum((1, 2)) > 0).mean()),
                 'a2_pos_set_frac': float((np.abs(pos2).sum((1, 2)) > 0).mean()),
                 'a1_pleasure_nonzero': None if p1 is None else float((p1 != 0).mean())}
            bad += r['empty_eps']
            lines.append(f"{os.path.relpath(path, _PI_DIR)} [{sp}] " + ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()))
    return lines, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*')
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    if a.names:
        paths = [os.path.join(_PI_DIR, 'data', 'data', n, 'data.h5') for n in a.names]
    else:
        paths = sorted(glob.glob(os.path.join(_PI_DIR, 'data', 'data', 'v15*', 'data.h5')))
        if a.all:
            paths += sorted(glob.glob(os.path.join(_PI_DIR, 'data', 'result', 'v15', 'datacheck', '*.h5')))
    total = 0
    for p in paths:
        lines, bad = check(p)
        total += bad
        for line in lines:
            print(line, flush=True)
    print('EMPTY EPISODES TOTAL:', total)
    sys.exit(1 if total else 0)


if __name__ == '__main__':
    main()
