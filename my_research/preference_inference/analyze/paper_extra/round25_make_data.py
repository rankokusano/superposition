"""
Paper round 25 Sec.1.1: combine the parts written by simulation/collect_data_round24.py into one stage-II style file
data/data/paper_r25_swap50/data.h5 (groups 'train' 3000 / 'test' 300, keys as v11_s2_pref plus swap_frame; no
compression, as v11_s2_pref). Parts: data/result/paper_extra/Round25/data_parts/{train,test}_part<i>.h5.
Settings: train key 'paper_round25_swap50', test key 'paper_round25_swap50_test'; p_swap 0.5; swap frame uniform
20-80; A-2 p_rule 0.5; liked colour and layout 1/2 per episode (the stage-II procedure).
Also prints the share of swap episodes, of each liked colour and of each layout per split.
"""

import glob
import os

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI = os.path.dirname(os.path.dirname(_HERE))
PARTS = os.path.join(_PI, 'data', 'result', 'paper_extra', 'Round25', 'data_parts')
OUT = os.path.join(_PI, 'data', 'data', 'paper_r25_swap50', 'data.h5')
KEYS = ('self_vision', 'other_vision', 'self_motion', 'self_position', 'other_motion', 'other_position', 'a1_source',
        'a2_source', 'layout_t', 'layout', 'a2_liked', 'swap_frame')


def main():
    assert not os.path.exists(OUT), OUT
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with h5py.File(OUT, 'w') as f:
        f.attrs['generator'] = 'simulation/collect_data_round24.py + analyze/paper_extra/round25_make_data.py'
        for split in ('train', 'test'):
            fs = sorted(glob.glob(os.path.join(PARTS, f'{split}_part*.h5')), key=lambda s: int(s.split('part')[-1][:-3]))
            hs = [h5py.File(p, 'r') for p in fs]
            idx = np.concatenate([h['idx'][()] for h in hs])
            assert np.array_equal(idx, np.arange(len(idx))), split
            g = f.create_group(split)
            for k in KEYS:
                g.create_dataset(k, data=np.concatenate([h[k][()] for h in hs]))
            for k in ('key', 'p_rule', 'swap_t', 'p_swap'):
                g.attrs[k] = hs[0].attrs[k]
            sw = g['swap_frame'][()]
            print(split, len(idx), 'swap share', float((sw >= 0).mean()), 'swap frame range', int(sw[sw >= 0].min()),
                  int(sw[sw >= 0].max()), 'red share', float(g['a2_liked'][()].mean()), 'layout1 share',
                  float(g['layout'][()].mean()))
    print('saved', OUT)


if __name__ == '__main__':
    main()
