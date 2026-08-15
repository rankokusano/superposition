"""
v4 R3-A follow-up (2026-08-14): does A-2's RL-driven reactivity to A-1's
appearance create a TEMPORAL (lagged) dependence between the two agents,
even though the SIMULTANEOUS position correlation was already measured
as near-zero? This would explain how process-1 (fed only self_vision)
could end up encoding A-2's position (h1->other = 0.582) despite never
directly observing A-2's real position.

Three checks, all on R2 data (r2_a1random_a2rl):
  1. how often A-1's avatar is actually visible in A-2's own captured
     vision (green/red-independent: just "is a non-background, non-floor
     object in frame" -- approximated by non-gray pixel fraction, since
     A-1's avatar is white/gray and would only show as brightness
     deviation against the textured floor/landmarks)
  2. Ridge R^2 for self_position(t) -> other_position(t+k), k=1,2,5,10
  3. Ridge R^2 for self_motion(t) -> other_motion(t+k), k=1,2,5,10

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/check_a1a2_temporal_dependence.py
"""
import json
import os

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
SAVE_DIR = 'data/result/baseline_v4'
LAGS = [1, 2, 5, 10]


def ridge_r2(X, Y):
    reg = Ridge(alpha=1.0)
    reg.fit(X, Y)
    pred = reg.predict(X)
    return float(r2_score(Y, pred))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    with h5py.File(DATA_H5, 'r') as f:
        sp = f['train/self_position'][()]   # (N, T, 2)
        op = f['train/other_position'][()]  # (N, T, 2)
        sm = f['train/self_motion'][()]     # (N, T, 2)
        om = f['train/other_motion'][()]    # (N, T, 2)

    N, T, _ = sp.shape
    print(f'N={N} T={T}')

    results = {'lagged_position_r2': {}, 'lagged_motion_r2': {}}

    print('\n=== Ridge R^2: self_position(t) -> other_position(t+k) ===')
    for k in LAGS:
        X = sp[:, :T - k].reshape(-1, 2)
        Y = op[:, k:].reshape(-1, 2)
        r2 = ridge_r2(X, Y)
        results['lagged_position_r2'][k] = r2
        print(f'  k={k:2d}: R^2 = {r2:.4f}')

    print('\n=== Ridge R^2: self_motion(t) -> other_motion(t+k) ===')
    for k in LAGS:
        X = sm[:, :T - k].reshape(-1, 2)
        Y = om[:, k:].reshape(-1, 2)
        r2 = ridge_r2(X, Y)
        results['lagged_motion_r2'][k] = r2
        print(f'  k={k:2d}: R^2 = {r2:.4f}')

    # sanity: also same-timestep (k=0) for reference, already measured before but recompute here for a self-contained record
    r2_pos0 = ridge_r2(sp.reshape(-1, 2), op.reshape(-1, 2))
    r2_mot0 = ridge_r2(sm.reshape(-1, 2), om.reshape(-1, 2))
    results['lagged_position_r2'][0] = r2_pos0
    results['lagged_motion_r2'][0] = r2_mot0
    print(f'\n(reference) k=0 position R^2={r2_pos0:.4f}  motion R^2={r2_mot0:.4f}')

    out_path = os.path.join(SAVE_DIR, 'r3a_temporal_dependence_check.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
