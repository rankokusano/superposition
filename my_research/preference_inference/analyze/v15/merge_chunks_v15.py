"""
Generic chunk merge for v15 datasets (docs/v15_experiment_log.md Sec.15).

    python analyze/v15/merge_chunks_v15.py --name v15_s2_c2_p80_half_6000 \
        --parts train:0:750:chunk ... test:0:300:chunk [train:0:3000:file=<other dataset>] \
        --check <check_a.h5>,<check_b.h5>,<split>:<start> [--first1000] [--delete_chunks]

Each part is <split>:<start>:<n>:<source>, source 'chunk' = data/data/<name>/chunks/<split>_<start>.h5 (its
episode_start must equal <start>), or 'file=<dataset>' = episodes start..start+n-1 of that dataset's <split>.
Parts are written in the given order per split. --check: two separate generations of the same episodes
(check_a, check_b) must be bit-identical to each other and to the merged episodes at <split>:<start>.
--first1000 also writes data/data/<name minus _NNNN>_1000t/data.h5 = train 0..999 + the whole test split.
"""

import argparse
import os
import shutil

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(_PI_DIR, 'data', 'data')


def src_of(name, split, start, source):
    if source == 'chunk':
        return os.path.join(D, name, 'chunks', f'{split}_{start}.h5'), split, 0, True
    assert source.startswith('file='), source
    return os.path.join(D, source[5:], 'data.h5'), split, start, False


def write(out, attrs, parts, name):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with h5py.File(out, 'w') as fw:
        for k, v in attrs.items():
            fw.attrs[k] = v
        fw.attrs['dataset'] = name
        fw.attrs['episode_start'] = 0
        for split, srcs in parts.items():
            n_tot = sum(p[-1] for p in srcs)
            with h5py.File(srcs[0][0], 'r') as f0:
                g0 = f0[srcs[0][1]]
                ds = {k: fw.create_dataset(f'{split}/{k}', (n_tot,) + g0[k].shape[1:], dtype=g0[k].dtype) for k in g0}
            pos = 0
            for path, grp, off, is_chunk, start, n in srcs:
                with h5py.File(path, 'r') as f:
                    if is_chunk:
                        assert f.attrs['episode_start'] == start and f[grp]['self_vision'].shape[0] == n, path
                    for k in ds:
                        ds[k][pos:pos + n] = f[grp][k][off:off + n]
                pos += n
            print(name, split, n_tot, 'episodes', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--parts', nargs='+', required=True)
    ap.add_argument('--check', required=True)
    ap.add_argument('--first1000', action='store_true')
    ap.add_argument('--delete_chunks', action='store_true')
    a = ap.parse_args()
    parts, attrs = {}, None
    for p in a.parts:
        split, start, n, source = p.split(':', 3)
        path, grp, off, is_chunk = src_of(a.name, split, int(start), source)
        parts.setdefault(split, []).append((path, grp, off, is_chunk, int(start), int(n)))
        if is_chunk and attrs is None:
            with h5py.File(path, 'r') as f:
                attrs = dict(f.attrs)
    ca, cb, where = a.check.split(',')
    csplit, cstart = where.split(':')
    out = os.path.join(D, a.name, 'data.h5')
    write(out, attrs, parts, a.name)
    with h5py.File(ca, 'r') as fa, h5py.File(cb, 'r') as fb, h5py.File(out, 'r') as fo:
        n = fa['train']['self_vision'].shape[0]
        s = int(cstart)
        bad_ab = [k for k in fa['train'] if not np.array_equal(fa['train'][k][:], fb['train'][k][:])]
        bad_m = [k for k in fa['train'] if not np.array_equal(fa['train'][k][:], fo[csplit][k][s:s + n])]
    print(f'check ({n} episodes at {csplit}:{s}) generated twice identical:', not bad_ab, bad_ab or '',
          '| == merged:', not bad_m, bad_m or '', flush=True)
    if bad_ab or bad_m:
        os.remove(out)
        raise SystemExit('reproducibility check failed; merged file removed')
    if a.first1000:
        base = a.name.rsplit('_', 1)[0]
        p1 = {'train': [(out, 'train', 0, False, 0, 1000)], 'test': [(out, 'test', 0, False, 0, parts['test'][0][-1])]}
        write(os.path.join(D, f'{base}_1000t', 'data.h5'), attrs, p1, f'{base}_1000t')
    if a.delete_chunks:
        shutil.rmtree(os.path.join(D, a.name, 'chunks'))
        print('chunks deleted', flush=True)


if __name__ == '__main__':
    main()
