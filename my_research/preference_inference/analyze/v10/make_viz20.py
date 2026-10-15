"""
Make a viz20 dataset (train group only, first N episodes) by h5 slicing, as
r2_a1random_a2rl_viz20 was made from r2_a1random_a2rl (verified identical to
its first 20 train episodes). With --per_layout K, takes the first K
episodes of each layout instead (B2: v10_swapmix_r2_viz20).
"""

import argparse
import os

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--per_layout', type=int, default=None)
    args = ap.parse_args()

    with h5py.File(args.src, 'r') as fs:
        g = fs['train']
        if args.per_layout is None:
            idx = np.arange(args.n)
        else:
            lay = g['layout'][()]
            idx = np.sort(np.concatenate([np.where(lay == c)[0][:args.per_layout] for c in np.unique(lay)]))
        os.makedirs(os.path.dirname(args.dst), exist_ok=True)
        with h5py.File(args.dst, 'w') as fd:
            for k, v in fs.attrs.items():
                fd.attrs[k] = v
            fd.attrs['viz_source'] = args.src
            fd.attrs['viz_train_indices'] = idx
            for k in g.keys():
                fd.create_dataset('train/' + k, data=g[k][idx])
    print('wrote', args.dst, 'episodes', idx.tolist())


if __name__ == '__main__':
    main()
