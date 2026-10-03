"""
S5 result analysis (docs/v6_experiment_log.md Sec.38.2, pre-registered
before S5 was run). Reads the saved.h5 produced by
`test.py --save_targets r_hat q2_hat self_position other_position state`
(e.g. analyze/run_eval_v4v5.sh's step 1, or the ad-hoc 5-checkpoint
"r_hat_trend" test_name used for the self-projection-over-training check)
and computes every pre-registered S5 metric for ONE epoch:

  1. Self-projection (Sec.38.2 (2)): mean r_hat vector's cosine similarity
     to A-1's true r (+1,-1,0,0) vs A-2's true r (-1,+1,0,0). Which is
     closer tells us whether VE' is projecting A-1's own reward onto the
     other agent (self-projection) or genuinely estimating A-2's.
  2. Two-stage judgment (Sec.38.2 (1)): the same cosine similarities,
     binned by A-2's distance to Green (bins fixed since Sec.27:
     <5, 5-10, 10-15, >=15), to see whether the overall signal (likely
     weak/absent, given S4's own near-bin dilution) strengthens at range
     -- mirroring S4's own fallback structure.
  3. Direction match (Sec.3 original S5 criterion): does Q_hat2's peak
     probe-direction point toward Green from A-2's actual position?
  4. Q_hat2 -> position R^2 for BOTH A-2's and A-1's position (Sec.3:
     "Q_hat2->A-2 position > Q_hat2->A-1 position", reversed from v4).
  5. h2 (process-2 SM output) -> self/other position R^2, continuing the
     v4 "position priority" pathology watch (Sec.38.2 (4)).

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/analyze_r_hat_v6.py --saved_h5 <path> --epoch 400 --mode eval --label v6_s5_ve_ep400
"""
import argparse
import json
import os

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

SAVE_DIR = 'data/result/v6_baseline'
GREEN_POS = np.array([-9.0, -9.0])
A1_TRUE_R = np.array([1.0, -1.0, 0.0, 0.0])
A2_TRUE_R = np.array([-1.0, 1.0, 0.0, 0.0])
BIN_EDGES = [0.0, 5.0, 10.0, 15.0, 1e9]
BIN_LABELS = ['<5', '5-10', '10-15', '>=15']
K = 8
PROBE_ANGLES = np.array([2 * np.pi * i / K for i in range(K)])
PROBE_ACTIONS = np.stack([np.cos(PROBE_ANGLES), np.sin(PROBE_ANGLES)], axis=1)  # (8,2)


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def r2(X, Y):
    reg = Ridge(alpha=1.0)
    reg.fit(X, Y)
    return float(r2_score(Y, reg.predict(X)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--saved_h5', required=True)
    ap.add_argument('--epoch', type=int, required=True)
    ap.add_argument('--mode', default='eval')
    ap.add_argument('--label', required=True)
    args = ap.parse_args()

    with h5py.File(args.saved_h5, 'r') as f:
        g = f[f'{args.epoch:05d}/{args.mode}']
        r_hat = g['r_hat/prediction'][:]       # (N,T,4)
        q2_hat = g['q2_hat/prediction'][:]     # (N,T,8)
        sp = g['self_position/truth'][:]       # (N,T,2)
        op = g['other_position/truth'][:]      # (N,T,2)
        h2 = g['state/other/hidden'][:]        # (N,T,128)

    N, T = r_hat.shape[:2]
    r_hat_flat = r_hat.reshape(-1, 4)
    q2_flat = q2_hat.reshape(-1, K)
    sp_flat = sp.reshape(-1, 2)
    op_flat = op.reshape(-1, 2)
    h2_flat = h2.reshape(-1, 128)

    result = {'label': args.label, 'epoch': args.epoch, 'mode': args.mode, 'n_frames': int(len(r_hat_flat))}

    # --- 1. self-projection: overall ---
    r_hat_mean = r_hat_flat.mean(axis=0)
    sim_a1 = cos_sim(r_hat_mean, A1_TRUE_R)
    sim_a2 = cos_sim(r_hat_mean, A2_TRUE_R)
    result['r_hat_mean'] = r_hat_mean.tolist()
    result['self_projection'] = {
        'cos_sim_to_A1_true': sim_a1, 'cos_sim_to_A2_true': sim_a2,
        'closer_to': 'A1 (self-projection)' if sim_a1 > sim_a2 else 'A2 (other-understanding)',
    }
    print(f'=== Self-projection (overall, {args.mode}, epoch {args.epoch}) ===')
    print(f'  mean r_hat = {r_hat_mean}')
    print(f'  cos_sim to A1_true(+1,-1,0,0) = {sim_a1:+.4f}')
    print(f'  cos_sim to A2_true(-1,+1,0,0) = {sim_a2:+.4f}')
    print(f'  closer to: {result["self_projection"]["closer_to"]}')

    # --- 2. distance-binned self-projection (two-stage judgment) ---
    dist_flat = np.linalg.norm(GREEN_POS[None, :] - op_flat, axis=1)
    dist_bins = {}
    print('\n=== Distance-binned self-projection ===')
    for lo, hi, lbl in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
        m = (dist_flat >= lo) & (dist_flat < hi)
        n = int(m.sum())
        if n == 0:
            dist_bins[lbl] = {'n': 0}
            continue
        rm = r_hat_flat[m].mean(axis=0)
        sa1, sa2 = cos_sim(rm, A1_TRUE_R), cos_sim(rm, A2_TRUE_R)
        dist_bins[lbl] = {'n': n, 'r_hat_mean': rm.tolist(),
                           'cos_sim_to_A1_true': sa1, 'cos_sim_to_A2_true': sa2}
        print(f'  {lbl:6s} (n={n:6d}): cos_sim(A1)={sa1:+.4f}  cos_sim(A2)={sa2:+.4f}  '
              f'closer_to={"A1" if sa1 > sa2 else "A2"}')
    result['distance_binned_self_projection'] = dist_bins

    # --- 3. direction match (Q_hat2 peak direction vs true direction to Green) ---
    to_green = GREEN_POS[None, :] - op_flat
    to_green_norm = to_green / (np.linalg.norm(to_green, axis=1, keepdims=True) + 1e-8)
    argmax_idx = q2_flat.argmax(axis=1)
    argmax_dir = PROBE_ACTIONS[argmax_idx]
    dir_cos_sim = (argmax_dir * to_green_norm).sum(axis=1)
    angle_err = np.degrees(np.arccos(np.clip(dir_cos_sim, -1, 1)))
    result['direction_match'] = {
        'mean_cos_sim': float(dir_cos_sim.mean()),
        'mean_angle_err_deg': float(angle_err.mean()),
        'frac_within_45deg': float((angle_err < 45).mean()),
    }
    print(f'\n=== Direction match (Q_hat2 peak dir vs true dir to Green) ===')
    print(f'  mean cos_sim = {dir_cos_sim.mean():+.4f}  (chance=0, perfect=1)')
    print(f'  frac within 45deg = {(angle_err < 45).mean()*100:.2f}%  (chance ~25%)')

    # --- 3b. distance-binned direction match (two-stage judgment, applied
    # to the direction metric too -- the most behaviourally meaningful one) ---
    dir_dist_bins = {}
    print('\n=== Distance-binned direction match ===')
    for lo, hi, lbl in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
        m = (dist_flat >= lo) & (dist_flat < hi)
        n = int(m.sum())
        if n == 0:
            dir_dist_bins[lbl] = {'n': 0}
            continue
        mc, mw = float(dir_cos_sim[m].mean()), float((angle_err[m] < 45).mean())
        dir_dist_bins[lbl] = {'n': n, 'mean_cos_sim': mc, 'frac_within_45deg': mw}
        print(f'  {lbl:6s} (n={n:6d}): mean_cos_sim={mc:+.4f}  within45={mw*100:.2f}%')
    result['distance_binned_direction_match'] = dir_dist_bins

    # --- 4. Q_hat2 -> position R^2 (A-2 vs A-1), overall AND distance-binned ---
    q2_to_a2 = r2(q2_flat, op_flat)
    q2_to_a1 = r2(q2_flat, sp_flat)
    result['q2_hat_r2'] = {'to_A2_other_position': q2_to_a2, 'to_A1_self_position': q2_to_a1}
    print(f'\n=== Q_hat2 -> position R^2 ===')
    print(f'  Q_hat2 -> A-2 (other) position = {q2_to_a2:.4f}')
    print(f'  Q_hat2 -> A-1 (self)  position = {q2_to_a1:.4f}')
    print(f'  A-2 > A-1 ? {"YES (correct direction)" if q2_to_a2 > q2_to_a1 else "NO (v4-style reversal)"}')

    q2_dist_bins = {}
    print('\n=== Distance-binned Q_hat2 -> position R^2 (within-bin comparison only; '
          'cross-bin magnitude is a range-restriction artifact, same caveat as SS36.4) ===')
    for lo, hi, lbl in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
        m = (dist_flat >= lo) & (dist_flat < hi)
        n = int(m.sum())
        if n <= 20:
            q2_dist_bins[lbl] = {'n': n}
            continue
        ra2, ra1 = r2(q2_flat[m], op_flat[m]), r2(q2_flat[m], sp_flat[m])
        q2_dist_bins[lbl] = {'n': n, 'to_A2_other_position': ra2, 'to_A1_self_position': ra1}
        print(f'  {lbl:6s} (n={n:6d}): Q_hat2->A2={ra2:.4f}  Q_hat2->A1={ra1:.4f}')
    result['distance_binned_q2_hat_r2'] = q2_dist_bins

    # --- 5. h2 (process-2 SM output) -> position R^2 (pathology watch) ---
    h2_to_self = r2(h2_flat, sp_flat)
    h2_to_other = r2(h2_flat, op_flat)
    result['h2_r2'] = {'h2_to_self': h2_to_self, 'h2_to_other': h2_to_other}
    print(f'\n=== h2 -> position R^2 (v4 "position priority" pathology watch) ===')
    print(f'  h2 -> self  = {h2_to_self:.4f}')
    print(f'  h2 -> other = {h2_to_other:.4f}')

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{args.label}_r_hat_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
