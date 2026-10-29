"""
Paper round 11 E1 training entry (docs/paper_extra_log.md Sec.35.4): train_paper_c.py (signed-pleasure model,
loader exp/loader_v14) with one more data name: 'paper_s1_mixed50_signrand' = the experienced-50% data v11_s1_mixed
read unchanged + a1_liked_t from data/data/paper_s1_mixed50_signrand/signs.h5 (the same flipped episodes as the 80%
signrand condition; per-episode code repeated over the steps), built exactly as train_paper_c.py builds the 80% one.
Every other data name goes to train_paper_c.py's loader. Nothing in v11-v15 or train_paper_c.py is changed.
Run from my_research/preference_inference with train_v12.py's arguments, e.g.
    python analyze/paper_extra/train_paper_e1.py --exp_config paper_extra/E1_signrand50/s1_signrand50_mse --seed 0 \
        --device cuda:0 --cudnn_deterministic --data_load_memory
"""
import os
import runpy
import sys

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PI_DIR = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PI_DIR)
sys.path.insert(0, _HERE)

import train_paper_c  # noqa  (registers the v14 classes and routes util.gen_data_loader)
import util  # noqa
from exp import loader_v14  # noqa

NAME = 'paper_s1_mixed50_signrand'
_c_loader = train_paper_c.gen_data_loader


def gen_data_loader(data_name, batch_size, data_load_memory, device):
    if data_name != NAME:
        return _c_loader(data_name, batch_size, data_load_memory, device)
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
