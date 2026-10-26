"""
Paper extra C training entry (docs/paper_extra_log.md Sec.1.4, 4): train_v14.py (signed-pleasure model, loader
exp/loader_v14 that passes a1_liked_t) with one more data name: 'paper_s1_mixed80_signrand' = the source data
v13_s1_mixed80 read unchanged + a1_liked_t from data/data/paper_s1_mixed80_signrand/signs.h5 (per-episode code
repeated over the 101 steps). 'paper_s1_mixed80_pzero' (log Sec.14) = v13_s1_mixed80 + a1_liked_t = 0.5 (float32)
everywhere, so the signed model's sign 2 * 0.5 - 1 = 0 makes process-1's pleasure input zero at every step (the hold
keeps zeros). Other data names behave exactly as in train_v14.py. Nothing in v12-v14 is changed.
Run from my_research/preference_inference with train_v12.py's arguments, e.g.
    python analyze/paper_extra/train_paper_c.py --exp_config paper_extra/C_signrand/s1_signrand_mse --seed 0 --device cuda:0 \
        --cudnn_deterministic --data_load_memory
"""
import os
import runpy
import sys

import h5py
import numpy as np

_PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PI_DIR)

import train_v14  # noqa  (registers the v14 classes, routes util.gen_data_loader to exp/loader_v14)
import util  # noqa
from exp import loader_v14  # noqa

NAME = 'paper_s1_mixed80_signrand'
NAME_ZERO = 'paper_s1_mixed80_pzero'
SRC_ZERO = 'v13_s1_mixed80'


def gen_data_loader(data_name, batch_size, data_load_memory, device):
    if data_name == NAME_ZERO:
        data = util.load_data(util.gen_data_filename(SRC_ZERO), data_load_memory)
        for k in data:
            data[k][loader_v14.KEY] = np.full(data[k]['self_position'].shape[:2], 0.5, np.float32)
        print(f'{NAME_ZERO}: {SRC_ZERO} + a1_liked_t = 0.5 (pleasure sign 0: P = 0 at every step)')
        return (loader_v14.DataLoaderLiked(data['train'], batch_size, True, device),
                loader_v14.DataLoaderLiked(data['train'], batch_size, False, device),
                loader_v14.DataLoaderLiked(data['test'], batch_size, False, device))
    if data_name != NAME:
        return loader_v14.gen_data_loader(data_name, batch_size, data_load_memory, device)
    with h5py.File(os.path.join(os.path.dirname(util.gen_data_filename(NAME)), 'signs.h5'), 'r') as f:
        src = f.attrs['source']
        liked = {k: f[k]['a1_liked'][()] for k in f}
    data = util.load_data(util.gen_data_filename(src), data_load_memory)
    for k, v in liked.items():
        assert data[k]['layout'].shape[0] == v.shape[0]
        data[k][loader_v14.KEY] = np.repeat(v[:, None], data[k]['self_position'].shape[1], 1)
    print(f'{NAME}: {src} + a1_liked_t (flipped train {int((liked["train"] == 0).sum())}/{len(liked["train"])})')
    return (loader_v14.DataLoaderLiked(data['train'], batch_size, True, device),
            loader_v14.DataLoaderLiked(data['train'], batch_size, False, device),
            loader_v14.DataLoaderLiked(data['test'], batch_size, False, device))


util.gen_data_loader = gen_data_loader

if __name__ == '__main__':
    sys.argv[0] = os.path.join(_PI_DIR, 'train_v12.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
