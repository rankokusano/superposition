"""
v11 dataset subsets by h5 slicing (docs/v11_experiment_log.md Sec.4.2, 4.14).

  --mode viz20      train only: the first 10 episodes of each layout
                    (same construction as v10_swapmix_r2_viz20). Because the
                    layout sequence is shared, both conditions get the same
                    episode indices (paired).
  --mode first1000  train = first 1000 train episodes, test = full test
                    (trial 6.2).
Attributes are copied; the source and the selected indices are recorded.
"""

import argparse
import os

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(_PI_DIR, 'data', 'data')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='dataset name, e.g. v11_s1_random')
    ap.add_argument('--mode', required=True, choices=['viz20', 'first1000'])
    args = ap.parse_args()

    src = os.path.join(DATA, args.src, 'data.h5')
    dst_name = f'{args.src}_{args.mode}'
    dst = os.path.join(DATA, dst_name, 'data.h5')
    assert args.src.startswith('v11_') and not os.path.exists(dst), dst
    os.makedirs(os.path.dirname(dst))
    with h5py.File(src, 'r') as fs, h5py.File(dst, 'w') as fd:
        for k, v in fs.attrs.items():
            fd.attrs[k] = v
        fd.attrs['subset_source'] = args.src
        fd.attrs['subset_mode'] = args.mode
        g = fs['train']
        if args.mode == 'viz20':
            lay = g['layout'][()]
            idx = np.sort(np.concatenate([np.where(lay == c)[0][:10] for c in (0, 1)]))
            splits = {'train': idx}
        else:
            splits = {'train': np.arange(1000), 'test': np.arange(fs['test']['layout'].shape[0])}
        for split, idx in splits.items():
            fd.attrs[f'subset_{split}_indices'] = idx
            for k in fs[split].keys():
                fd.create_dataset(f'{split}/{k}', data=fs[split][k][idx[0]:idx[-1] + 1][idx - idx[0]])
    print('wrote', dst, {s: (len(i), i[:20].tolist()) for s, i in splits.items()})


if __name__ == '__main__':
    main()
