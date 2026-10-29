"""
Paper round 11, E1 data (docs/paper_extra_log.md Sec.35.4): the experienced-50% stage-1 data v11_s1_mixed with the
pleasure sign flipped per episode, the flipped episodes being the SAME as in the 80% sign-randomised condition: the
same draw as simulation/make_signrand_paper_c.py (RandomState(md5('paper_c_signrand/a1_sign')).rand() < 0.5 per
episode over train (3000) then test (300)), checked to equal data/data/paper_s1_mixed80_signrand/signs.h5. Only A-1's
per-episode code is written, to data/data/paper_s1_mixed50_signrand/signs.h5 (<split>/a1_liked, 1 = sign kept, 0 =
flipped); analyze/paper_extra/train_paper_e1.py reads v11_s1_mixed unchanged and adds a1_liked_t.
Usage (inside Docker): python simulation/make_signrand_paper_e1.py
"""

import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_data_v11 as base  # noqa
import make_signrand_paper_c as c80  # noqa

NAME = 'paper_s1_mixed50_signrand'
SRC_NAME = 'v11_s1_mixed'
DST = os.path.join(base.DATA_ROOT, 'data', NAME, 'signs.h5')


def main():
    assert not os.path.exists(DST), DST
    rng = np.random.RandomState(base.get_seed(c80.SIGN_KEY))
    flip = rng.rand(sum(base.FULL.values())) < 0.5
    liked = (~flip).astype(np.uint8)
    with h5py.File(c80.DST, 'r') as f80:
        for split, a in (('train', liked[:base.FULL['train']]), ('test', liked[base.FULL['train']:])):
            assert np.array_equal(f80[split]['a1_liked'][()], a), f'{split}: differs from the 80% sign table'
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    rec = {'sign_key': c80.SIGN_KEY, 'source': SRC_NAME, 'same_as': os.path.relpath(c80.DST, base.DATA_ROOT)}
    with h5py.File(os.path.join(base.DATA_ROOT, 'data', SRC_NAME, 'data.h5'), 'r') as s, h5py.File(DST, 'w') as d:
        d.attrs['dataset'] = NAME
        d.attrs['source'] = SRC_NAME
        d.attrs['sign_key'] = c80.SIGN_KEY
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
