"""
Paper extra C data (docs/paper_extra_log.md Sec.1.4, 4): the experienced-80% stage-1 data v13_s1_mixed80 with the
pleasure sign flipped in a random half of the episodes. No simulation and no copy of the 2 GB source: only A-1's
per-episode "liked" code is written, to data/data/paper_s1_mixed80_signrand/signs.h5 (<split>/a1_liked, 1 = sign kept
(Red), 0 = flipped (Green)); analyze/paper_extra/train_paper_c.py reads v13_s1_mixed80 unchanged and adds
a1_liked_t for model_v14.SuperpositionNetworkMotionPleasureHoldSigned (as v14_s1_randompref, made by copying).
Flip draw: RandomState(md5('paper_c_signrand/a1_sign')).rand() < 0.5 per episode over train (3000) then test (300),
independent of the layout draw (md5('v11_s1/layout')).
Usage (inside Docker): python simulation/make_signrand_paper_c.py
"""

import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_data_v11 as base  # noqa

NAME = 'paper_s1_mixed80_signrand'
SRC_NAME = 'v13_s1_mixed80'
SIGN_KEY = 'paper_c_signrand/a1_sign'
DST = os.path.join(base.DATA_ROOT, 'data', NAME, 'signs.h5')


def main():
    assert not os.path.exists(DST), DST
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    rng = np.random.RandomState(base.get_seed(SIGN_KEY))
    flip = rng.rand(sum(base.FULL.values())) < 0.5
    liked = (~flip).astype(np.uint8)
    rec = {'sign_key': SIGN_KEY, 'source': SRC_NAME}
    with h5py.File(os.path.join(base.DATA_ROOT, 'data', SRC_NAME, 'data.h5'), 'r') as s, h5py.File(DST, 'w') as d:
        d.attrs['dataset'] = NAME
        d.attrs['source'] = SRC_NAME
        d.attrs['sign_key'] = SIGN_KEY
        d.attrs['a1_liked'] = '1 = pleasure sign kept (Red - Green), 0 = flipped (Green - Red); A-1 still approaches Red'
        i0 = 0
        for split, n in base.FULL.items():
            assert s[split]['layout'].shape[0] == n
            d.create_dataset(f'{split}/a1_liked', data=liked[i0:i0 + n])
            lay = s[split]['layout'][()]
            rec[split] = {'flipped_episodes': np.where(flip[i0:i0 + n])[0].tolist(),
                          'n_flipped': int(flip[i0:i0 + n].sum()), 'n': n,
                          'layout_x_flip_counts': {f'layout{L}_flip{int(F)}': int(((lay == L) & (flip[i0:i0 + n] == F)).sum())
                                                   for L in (0, 1) for F in (False, True)}}
            i0 += n
    with open(DST.replace('.h5', '.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    print('written', DST, {k: (v['n_flipped'], v['layout_x_flip_counts']) for k, v in rec.items() if isinstance(v, dict)})


if __name__ == '__main__':
    main()
