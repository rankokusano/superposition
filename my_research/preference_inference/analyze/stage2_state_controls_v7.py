"""
State-contribution controls for v7 stages 1-2 (docs/v7_experiment_log.md
§2.9). Added AFTER seeing the stage-2 results; the gates are unchanged.

Does the critic-based inference use the state at all, or does it reduce to
classifying the action's direction (Red is always to the north-west etc.)?
Action = A-2's true action in every condition.
  z1: state = uniform gray image (all pixels 0.5)
  z2: state = A-2's TRUE vision from another test episode at the same t
      (episodes permuted with RandomState(0), no fixed points)
  z3: no critic -- Q_geo(a, r) = cos(angle(a) - theta_r), theta_r = direction
      from the arena centre to the goal landmark (Red 135, Green 225, Cyan 45 deg)

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/stage2_state_controls_v7.py --git_commit <hash>
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, '/work/my_research/preference_inference/analyze')
import irl_common_v7 as C  # noqa
from irl_stage1_v7 import irl_by_t  # noqa

DATA_PATH = os.path.join(C.PI_ROOT, 'data', 'data', 'v7_a1random_a2goal3', 'data.h5')
SAVE_DIR = os.path.join(C.PI_ROOT, 'data', 'result', 'v7_irl', 'stage2')
THETA = np.radians([135.0, 225.0, 45.0])


def derangement(n, seed=0):
    rng = np.random.RandomState(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--git_commit', required=True)
    args = ap.parse_args()
    C.set_determinism(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f = h5py.File(DATA_PATH, 'r')
    ov = f['test/other_vision'][()]
    om = f['test/other_motion'][()]
    opos = f['test/other_position'][()]
    goal = f['test/goal_index'][()]
    N, T = om.shape[:2]
    bins = C.init_bins(np.linalg.norm(opos[:, 0] - C.GOAL_POS[goal], axis=-1))
    unit, small = C.unit_actions(om)
    use = ~small
    uniform3 = np.log(np.full(3, 1 / 3))
    critic = C.load_critic(device)
    out = dict(git_commit=args.git_commit, added_after_seeing_stage2=True,
               true_values_used='A-2 true action (all), A-2 true vision of OTHER episodes (z2), '
                                'true goal label (evaluation)')

    gray = np.full((1, *ov.shape[2:]), 0.5, dtype=np.float32)
    perm = derangement(N)
    ov_perm = ov[perm]
    out['z2_permutation_seed'] = 0
    conds = {
        'z1_gray': lambda i0, i1: np.repeat(gray, i1 - i0, axis=0),
        'z2_shuffled_true_vision': lambda i0, i1: ov_perm.reshape(N * T, *ov.shape[2:])[i0:i1].astype(np.float32) / 255.0,
    }
    for name, fn in conds.items():
        q64, qobs = C.compute_q(critic, fn, N * T, unit.reshape(N * T, 2), device)
        q64 = q64.reshape(N, T, len(C.CAND_R), C.K_DIRS)[:, :, :3]
        qobs = qobs.reshape(N, T, len(C.CAND_R))[:, :, :3]
        out[name], _ = irl_by_t(q64, qobs, use, C.BETA_MAIN, uniform3, goal, bins)
        k = q64.argmax(-1)
        out[name + '_argmax_deg_mode'] = {C.CAND_NAMES[j]: dict(
            deg=float(np.bincount(k[..., j].ravel()).argmax() * 360 / C.K_DIRS),
            frac=float(np.bincount(k[..., j].ravel()).max() / k[..., j].size)) for j in range(3)}
        print(f'{name:26s} ' + '  '.join(f't={t}:{out[name][t]["acc"]:.3f}' for t in C.T_EVAL))

    ang = np.arctan2(unit[..., 1], unit[..., 0])                               # (N, T)
    k_ang = 2 * np.pi * np.arange(C.K_DIRS) / C.K_DIRS
    q64g = np.cos(k_ang[None, None, None, :] - THETA[None, None, :, None]) * np.ones((N, T, 1, 1))
    qobsg = np.cos(ang[..., None] - THETA[None, None, :])
    out['z3_geometric_direction'], _ = irl_by_t(q64g, qobsg, use, C.BETA_MAIN, uniform3, goal, bins)
    print(f'{"z3_geometric_direction":26s} ' + '  '.join(
        f't={t}:{out["z3_geometric_direction"][t]["acc"]:.3f}' for t in C.T_EVAL))

    path = os.path.join(SAVE_DIR, 'stage2_state_controls.json')
    with open(path, 'w') as fp:
        json.dump(out, fp, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('Saved:', path)


if __name__ == '__main__':
    main()
