"""
v15 reproducibility check (docs/v15_experiment_log.md Sec.4.1): every array
and attribute of two generations must be bit-identical.
  * v15_s1_p80, v15_s2_c2_p80: full dataset vs datacheck/<name>_full_repeat.h5
  * every other dataset: datacheck/<name>_check_a.h5 vs _check_b.h5, and
    check_a vs the first 20 episodes of the full dataset

    python analyze/v15/check_repro_v15.py
"""

import os
import sys

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)
from simulation.collect_data_v15 import CHECK, DATA_ROOT, DATASET_CONFIGS  # noqa

FULL_REPEAT = ('v15_s1_p80', 'v15_s2_c2_p80')
SKIP_ATTRS = ('dataset',)


def compare(pa, pb, n=None):
    bad = []
    with h5py.File(pa, 'r') as fa, h5py.File(pb, 'r') as fb:
        for k in fa.attrs:
            if k not in SKIP_ATTRS and str(fa.attrs[k]) != str(fb.attrs.get(k)):
                bad.append(f'attr {k}')
        ga, gb = fa['train'], fb['train']
        if set(ga) != set(gb):
            bad.append('keys differ')
        for k in ga:
            a = ga[k][:n] if n else ga[k][:]
            b = gb[k][:n] if n else gb[k][:]
            if a.shape != b.shape or not np.array_equal(a, b):
                bad.append(k)
    return bad


def main():
    names = [k for k in DATASET_CONFIGS if '_check_' not in k]
    ok_all = True
    for name in names:
        full = os.path.join(DATA_ROOT, 'data', name, 'data.h5')
        if name in FULL_REPEAT:
            bad = compare(full, os.path.join(CHECK, f'{name}_full_repeat.h5'))
            print(f'{name}: full 1000 episodes, two generations identical: {not bad} {bad or ""}')
            ok_all &= not bad
        else:
            a, b = (os.path.join(CHECK, f'{name}_check_{r}.h5') for r in 'ab')
            bad_ab = compare(a, b)
            bad_full = compare(a, full, n=20)
            print(f'{name}: check a == b: {not bad_ab} {bad_ab or ""}; check a == first 20 of full: {not bad_full} {bad_full or ""}')
            ok_all &= not bad_ab and not bad_full
    print('ALL IDENTICAL' if ok_all else 'MISMATCH FOUND')


if __name__ == '__main__':
    main()
