"""
v15 stage-1 data with gray in half of the episodes (docs/v15_experiment_log.md Sec.12):
build data/data/v15_s1_p80_half_3000/data.h5 (train 3000 from chunks train_0/750/1500/2250, test 300 from test_0)
and data/data/v15_s1_p80_half_first1000/data.h5 (train = its first 1000 episodes; for the Fig.5 Decoders), after
checking that two separate generations of train episodes 0..19 (check_a, check_b) are bit-identical to each other
and to the merged train 0..19. Chunks are deleted only with --delete_chunks (check files are kept).

    python analyze/v15/merge_s1half_v15.py [--delete_chunks]
"""

import argparse
import os
import shutil

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(_PI_DIR, 'data', 'data')
NAME = 'v15_s1_p80_half_3000'
CH = os.path.join(D, NAME, 'chunks')
CHECK = os.path.join(_PI_DIR, 'data', 'result', 'v15', 'datacheck')
TRAIN = [(0, 750), (750, 750), (1500, 750), (2250, 750)]


def write(out, attrs, parts, name):
    with h5py.File(out, 'w') as fw:
        for k, v in attrs.items():
            fw.attrs[k] = v
        fw.attrs['dataset'] = name
        fw.attrs['episode_start'] = 0
        for split, srcs in parts.items():
            n_tot = sum(n for *_, n in srcs)
            with h5py.File(srcs[0][0], 'r') as f0:
                g0 = f0[srcs[0][1]]
                ds = {k: fw.create_dataset(f'{split}/{k}', (n_tot,) + g0[k].shape[1:], dtype=g0[k].dtype) for k in g0}
            pos = 0
            for path, grp, start, n in srcs:
                with h5py.File(path, 'r') as f:
                    assert f.attrs['episode_start'] == start and f[grp]['self_vision'].shape[0] >= n, path
                    for k in ds:
                        ds[k][pos:pos + n] = f[grp][k][:n]
                pos += n
            print(name, split, n_tot, 'episodes', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delete_chunks', action='store_true')
    a = ap.parse_args()
    ca, cb = (os.path.join(CHECK, f'v15_s1_p80_half_train0_check_{r}.h5') for r in 'ab')
    with h5py.File(ca, 'r') as fa, h5py.File(cb, 'r') as fb, h5py.File(os.path.join(CH, 'train_0.h5'), 'r') as f0:
        bad_ab = [k for k in fa['train'] if not np.array_equal(fa['train'][k][:], fb['train'][k][:])]
        bad_0 = [k for k in fa['train'] if not np.array_equal(fa['train'][k][:], f0['train'][k][:20])]
        attrs = dict(f0.attrs)
    print('train 0..19 generated twice identical:', not bad_ab, bad_ab or '', '| == chunk train_0:', not bad_0, bad_0 or '')
    assert not bad_ab and not bad_0
    attrs['composition'] = 'train 0-2999 and test 0-299 generated in chunks (key v15_s1, gray in half of the episodes)'
    parts = {'train': [(os.path.join(CH, f'train_{s}.h5'), 'train', s, n) for s, n in TRAIN],
             'test': [(os.path.join(CH, 'test_0.h5'), 'test', 0, 300)]}
    write(os.path.join(D, NAME, 'data.h5'), attrs, parts, NAME)
    os.makedirs(os.path.join(D, 'v15_s1_p80_half_first1000'), exist_ok=True)
    write(os.path.join(D, 'v15_s1_p80_half_first1000', 'data.h5'), attrs,
          {'train': [(os.path.join(CH, 'train_0.h5'), 'train', 0, 750), (os.path.join(CH, 'train_750.h5'), 'train', 750, 250)]},
          'v15_s1_p80_half_first1000')
    if a.delete_chunks:
        shutil.rmtree(CH)
        print('chunks deleted')


if __name__ == '__main__':
    main()
