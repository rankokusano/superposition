"""
v15 stage-1 data (docs/v15_experiment_log.md Sec.8): build data/data/v15_s1_p80_3000/data.h5 =
  train 0..999   copied from v15_s1_p80 (already verified bit-reproducible),
  train 1000..2999 from chunks train_1000/1500/2000/2500,
  test 0..299    from chunk test_0;
after checking that the regenerated train episodes 980..999 (chunk train_980) are bit-identical to
v15_s1_p80's. Attributes from v15_s1_p80 (dataset name updated). Chunks are deleted only with --delete_chunks.

    python analyze/v15/merge_s1x_v15.py [--delete_chunks]
"""

import argparse
import os
import shutil

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(_PI_DIR, 'data', 'data')
SRC = os.path.join(D, 'v15_s1_p80', 'data.h5')
OUT_DIR = os.path.join(D, 'v15_s1_p80_3000')
CH = os.path.join(OUT_DIR, 'chunks')
TRAIN_CHUNKS = (1000, 1500, 2000, 2500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delete_chunks', action='store_true')
    a = ap.parse_args()
    with h5py.File(SRC, 'r') as fs, h5py.File(os.path.join(CH, 'train_980.h5'), 'r') as fo:
        bad = [k for k in fo['train'] if not np.array_equal(fo['train'][k][:], fs['train'][k][980:1000])]
    print('overlap check (train 980..999 regenerated vs v15_s1_p80):', 'identical' if not bad else f'MISMATCH {bad}')
    assert not bad
    out = os.path.join(OUT_DIR, 'data.h5')
    with h5py.File(SRC, 'r') as fs, h5py.File(out, 'w') as fw:
        for k, v in fs.attrs.items():
            fw.attrs[k] = v
        fw.attrs['dataset'] = 'v15_s1_p80_3000'
        fw.attrs['episode_start'] = 0
        fw.attrs['composition'] = 'train 0-999 = v15_s1_p80; train 1000-2999 and test 0-299 generated in chunks (same key)'
        parts = {'train': [(SRC, 'train', 0, 1000)] + [(os.path.join(CH, f'train_{s}.h5'), 'train', s, 500) for s in TRAIN_CHUNKS],
                 'test': [(os.path.join(CH, 'test_0.h5'), 'test', 0, 300)]}
        for split, srcs in parts.items():
            n_tot = sum(n for *_, n in srcs)
            with h5py.File(srcs[0][0], 'r') as f0:
                g0 = f0[srcs[0][1]]
                ds = {k: fw.create_dataset(f'{split}/{k}', (n_tot,) + g0[k].shape[1:], dtype=g0[k].dtype) for k in g0}
            pos = 0
            for path, grp, start, n in srcs:
                with h5py.File(path, 'r') as f:
                    if path != SRC:
                        assert f.attrs['episode_start'] == start and f[grp]['self_vision'].shape[0] == n, path
                    for k in ds:
                        ds[k][pos:pos + n] = f[grp][k][:n]
                pos += n
            print(split, n_tot, 'episodes')
    print('saved', out)
    if a.delete_chunks:
        shutil.rmtree(CH)
        print('chunks deleted')


if __name__ == '__main__':
    main()
