"""
Combine multiple h5 datasets into one.

Usage (inside Docker /work):
    python my_research/preference_inference/simulation/combine_dataset.py \
        --targets new_self_rl_other_red new_self_rl_other_green new_self_rl_other_blue new_self_rl_other_yellow \
        --save_as new_self_rl_other_cycler
"""

import argparse
import os

import h5py
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('--targets', nargs='+', type=str, required=True)
parser.add_argument('--save_as', required=True)
args = parser.parse_args()

WORK_ROOT = '/work' if os.path.isdir('/work') else os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATA_ROOT = os.path.join(WORK_ROOT, 'my_research', 'preference_inference', 'data', 'data')

targets = []
for target in args.targets:
    f_target = os.path.join(DATA_ROOT, target, 'data.h5')
    with h5py.File(f_target, 'r') as f_h5:
        dct = {}
        for mode in f_h5.keys():
            dct[mode] = {}
            for modal in f_h5[mode].keys():
                dct[mode][modal] = f_h5[mode][modal][()]
        targets.append(dct)

save_dir = os.path.join(DATA_ROOT, args.save_as)
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'data.h5')

h5_dataset = h5py.File(save_path, 'w')

for mode in targets[0].keys():
    for modal in targets[0][mode].keys():
        data = []
        for target in targets:
            if mode in target and modal in target[mode]:
                data.append(target[mode][modal])
        if not data:
            continue
        combined = numpy.concatenate(data, axis=0)
        # Preserve dtype: landmark labels are int, others float
        dtype = combined.dtype
        ds = h5_dataset.create_dataset(f'{mode}/{modal}', combined.shape, dtype=dtype)
        ds[:] = combined

h5_dataset.close()
print(f"Saved combined dataset to {save_path}")
