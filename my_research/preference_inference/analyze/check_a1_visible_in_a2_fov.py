"""
v4 R3-A follow-up: how often is A-1's avatar actually visible in A-2's
own captured vision (other_vision)? A-1's avatar uses self.mtl (white,
Kd=1,1,1), which should render distinctly brighter than the floor/
landmark textures. Approximated as near-white pixel fraction exceeding
a threshold.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/check_a1_visible_in_a2_fov.py
"""
import json
import os

import h5py
import numpy as np

DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
SAVE_DIR = 'data/result/baseline_v4'
N_SAMPLE = 20000
WHITE_PIXEL_THRESHOLD = 20  # px count considered "A-1 visible"


def main():
    rng = np.random.RandomState(0)
    with h5py.File(DATA_H5, 'r') as f:
        n_data, seq_len, H, W, C = f['train/other_vision'].shape
        flat_n = n_data * seq_len
        idx = rng.choice(flat_n, size=min(N_SAMPLE, flat_n), replace=False)
        n_idx = idx // seq_len
        t_idx = idx % seq_len
        vision = np.zeros((len(idx), H, W, C), dtype=np.uint8)
        order = {}
        for i, (n, t) in enumerate(zip(n_idx, t_idx)):
            order.setdefault(int(n), []).append((i, int(t)))
        for n, items in order.items():
            ep_vision = f['train/other_vision'][n]
            for i, t in items:
                vision[i] = ep_vision[t]

    v = vision.astype(np.float32) / 255.0
    r, g, b = v[:, :, :, 0], v[:, :, :, 1], v[:, :, :, 2]
    white_mask = (r > 0.9) & (g > 0.9) & (b > 0.9)
    white_px_count = white_mask.reshape(len(v), -1).sum(axis=1)

    visible_frac = float((white_px_count > WHITE_PIXEL_THRESHOLD).mean())
    print(f'n_samples={len(v)}')
    print(f'white_px_count: mean={white_px_count.mean():.2f}  '
          f'median={np.median(white_px_count):.1f}  max={white_px_count.max()}')
    print(f'fraction of frames with >{WHITE_PIXEL_THRESHOLD} near-white px '
          f'(proxy for "A-1 visible"): {visible_frac*100:.2f}%')

    for thresh in [1, 5, 10, 20, 50]:
        frac = float((white_px_count > thresh).mean())
        print(f'  threshold={thresh:3d}px: {frac*100:6.2f}% of frames')

    result = {
        'n_samples': int(len(v)),
        'white_px_count_mean': float(white_px_count.mean()),
        'white_px_count_median': float(np.median(white_px_count)),
        'visible_fraction_at_20px': visible_frac,
        'visible_fraction_by_threshold': {
            str(t): float((white_px_count > t).mean()) for t in [1, 5, 10, 20, 50]
        },
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, 'a1_visibility_in_a2_fov.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
