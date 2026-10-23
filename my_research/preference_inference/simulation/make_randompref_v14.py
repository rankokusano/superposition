"""
v14-C no-experience version data (docs/v14_experiment_log.md Sec.24): v11_s1_random (A-1 walks randomly) copied
unchanged, plus A-1's per-episode liked colour from the SAME sequence as v14_s1_mixedpref
(collect_data_v14.episode_liked: RandomState(md5('v14c/a1_liked')) over the train+test order), as `a1_liked`
(0 Green, 1 Red) and `a1_liked_t` (per step). No simulation: the walk does not depend on the liked colour, only the
pleasure sign does (applied in model_v14.SuperpositionNetworkMotionPleasureHoldSigned).
-> data/data/v14_s1_randompref/data.h5. Usage (inside Docker): python simulation/make_randompref_v14.py
"""

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_data_v14 import base, episode_liked  # noqa

SRC = os.path.join(base.DATA_ROOT, 'data', 'v11_s1_random', 'data.h5')
DST = os.path.join(base.DATA_ROOT, 'data', 'v14_s1_randompref', 'data.h5')


def main():
    assert not os.path.exists(DST), DST
    os.makedirs(os.path.dirname(DST))
    liked = episode_liked(sum(base.FULL.values()))
    with h5py.File(SRC, 'r') as s, h5py.File(DST, 'w') as d:
        for k, v in s.attrs.items():
            d.attrs[k] = v
        d.attrs['dataset'] = 'v14_s1_randompref'
        d.attrs['a1_liked'] = 'per episode: dataset a1_liked (0 Green, 1 Red), same sequence as v14_s1_mixedpref'
        d.attrs['source'] = 'copy of v11_s1_random + a1_liked'
        i0 = 0
        for split, n in base.FULL.items():
            for k in s[split]:
                s.copy(s[split][k], d.require_group(split), name=k)
            assert s[split]['layout'].shape[0] == n
            d[split].create_dataset('a1_liked', data=liked[i0:i0 + n])
            d[split].create_dataset('a1_liked_t', data=np.repeat(liked[i0:i0 + n, None], 101, 1))
            i0 += n
    print('written', DST)


if __name__ == '__main__':
    main()
