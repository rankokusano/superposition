"""
v7 stage 1: upper bound -- A-2's TRUE vision as s and A-2's TRUE action as a
(docs/v7_experiment_log.md §2.2, §2.5-§2.7).

True values used in this stage:
  - inference: A-2's true vision (other_vision) and true action (other_motion)
  - diagnostics only (§2.7): A-2's true position (moving-frame selection,
    near/far split); the true goal label for the moving-frame selection
  - baseline only: A-2's true position (position classifier, train -> test)
  - evaluation only: the true goal label (goal_index)

Usage (inside Docker):
    cd /work/my_research/preference_inference
    python analyze/irl_stage1_v7.py --git_commit <hash> [--suffix _run2]
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/work/my_research/preference_inference/analyze')
import irl_common_v7 as C  # noqa

DATASET = 'v7_a1random_a2goal3'
DATA_PATH = os.path.join(C.PI_ROOT, 'data', 'data', DATASET, 'data.h5')
SAVE_DIR = os.path.join(C.PI_ROOT, 'data', 'result', 'v7_irl', 'stage1')
K_BINS = ((0, 1), (1, 5), (5, 10), (10, 20), (20, 10 ** 9))
K_BIN_NAMES = ('0', '1-4', '5-9', '10-19', '>=20')


def position_features(pos, t):
    p = pos[:, :t]
    return np.concatenate([p[:, 0], p[:, -1], p[:, -1] - p[:, 0], p.mean(axis=1)], axis=1)


def position_baseline(train_pos, train_goal, test_pos, test_goal, bins):
    out = {}
    for t in C.T_EVAL:
        sc = StandardScaler().fit(position_features(train_pos, t))
        clf = LogisticRegression(multi_class='multinomial', C=1.0, max_iter=1000)
        clf.fit(sc.transform(position_features(train_pos, t)), train_goal)
        proba = clf.predict_proba(sc.transform(position_features(test_pos, t)))
        out[t] = C.evaluate(proba, test_goal, bins, post=proba)
    return out


def irl_by_t(q64, qobs, mask, betas, log_prior, goal, bins):
    lp = C.log_posterior_all_t(q64, qobs, mask, betas, log_prior)
    post = np.exp(lp)
    return {t: C.evaluate(lp[:, t], goal, bins, post=post[:, t]) for t in C.T_EVAL}, post


def moving_only(q64, qobs, move_mask, goal, bins, label):
    """Posterior after all moving frames of each episode (frames 0..100)."""
    lp = C.log_posterior_all_t(q64, qobs, move_mask, C.BETA_MAIN, np.log(np.full(C.N_MAIN, 1 / C.N_MAIN)))
    final = lp[:, -1]
    k = move_mask.sum(axis=1)
    res = C.evaluate(final, goal, bins, post=np.exp(final))
    res['definition'] = label
    res['k_mean'] = float(k.mean())
    res['by_k'] = {}
    pred, _ = C.argmax_with_ties(final)
    for (lo, hi), name in zip(K_BINS, K_BIN_NAMES):
        m = (k >= lo) & (k < hi)
        res['by_k'][name] = dict(n=int(m.sum()), acc=float((pred[m] == goal[m]).mean()) if m.any() else None)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--git_commit', required=True)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()

    C.set_determinism(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f = h5py.File(DATA_PATH, 'r')
    ov = f['test/other_vision'][()]          # (N, T, H, W, 3) uint8
    om = f['test/other_motion'][()]          # (N, T, 2)
    opos = f['test/other_position'][()]      # (N, T, 2)
    goal = f['test/goal_index'][()]
    train_pos = f['train/other_position'][()]
    train_goal = f['train/goal_index'][()]
    N, T = om.shape[:2]
    print(f'test: N={N} T={T}, goals={np.bincount(goal)}')

    dist_goal = np.linalg.norm(opos - C.GOAL_POS[goal][:, None], axis=-1)            # (N, T) true goal
    dist_all = np.linalg.norm(opos[:, :, None] - C.GOAL_POS[None, None], axis=-1)    # (N, T, 3)
    bins = C.init_bins(dist_goal[:, 0])

    critic = C.load_critic(device)
    eq_diff = C.check_trunk_head_equivalence(critic, ov, device)
    print(f'trunk/head split vs CriticLSTM.forward: max|diff|={eq_diff:.2e}')

    unit, small = C.unit_actions(om)
    flat_ov = ov.reshape(N * T, *ov.shape[2:])
    q64, qobs = C.compute_q(critic, lambda i0, i1: flat_ov[i0:i1].astype(np.float32) / 255.0,
                            N * T, unit.reshape(N * T, 2), device)
    q64 = q64.reshape(N, T, len(C.CAND_R), C.K_DIRS)
    qobs = qobs.reshape(N, T, len(C.CAND_R))
    use = ~small
    print(f'frames with |a|<{C.SMALL_ACTION} (excluded): {small.mean():.4f}')

    os.makedirs(SAVE_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(SAVE_DIR, f'q_stage1{args.suffix}.npz'), q64=q64, qobs=qobs)

    uniform3 = np.log(np.full(C.N_MAIN, 1 / C.N_MAIN))
    m3 = (q64[:, :, :C.N_MAIN], qobs[:, :, :C.N_MAIN])
    res = dict(
        experiment='v7_s1_true_state_true_action', git_commit=args.git_commit,
        dataset=DATASET, split='test', critic=C.CRITIC_PATH, seeds=dict(torch=0, numpy=0, tie_break=0),
        true_values_used=dict(
            inference='A-2 true vision (other_vision) + A-2 true action (other_motion)',
            diagnostics='A-2 true position and true goal label (moving-frame selection, near/far split)',
            baseline='A-2 true position (position classifier, train split -> test split)',
            evaluation='true goal label'),
        n_test=int(N), goal_counts=np.bincount(goal).tolist(), K=C.K_DIRS,
        beta_main=list(C.BETA_MAIN), beta_ext=list(C.BETA_EXT),
        trunk_head_max_abs_diff=eq_diff, frac_small_action=float(small.mean()),
        init_bin_counts={n: int((bins == b).sum()) for b, n in enumerate(C.BIN_NAMES)},
    )

    # Main and secondary IRL variants.
    res['irl_main'], post_main = irl_by_t(*m3, use, C.BETA_MAIN, uniform3, goal, bins)
    res['irl_beta_ext'], _ = irl_by_t(*m3, use, C.BETA_EXT, uniform3, goal, bins)
    res['irl_4cand_with_blue'], _ = irl_by_t(q64, qobs, use, C.BETA_MAIN, np.log(np.full(4, 0.25)), goal, bins)
    res['self_projection_prior'], _ = irl_by_t(*m3, use, C.BETA_MAIN, np.log(np.array([0.5, 0.25, 0.25])),
                                               goal, bins)
    res['self_projection_always_r1'] = C.evaluate(
        np.tile(np.eye(C.N_MAIN)[C.R1_INDEX], (N, 1)), goal, bins)

    # Colour-fraction baseline on the same (true) vision.
    frac, no_pix = C.landmark_fraction_array(flat_ov.astype(np.float32) / 255.0)
    frac = frac.reshape(N, T, 4)
    res['frac_frames_no_colour_pixel'] = float(no_pix.mean())
    cs = C.color_scores(frac)
    res['colour_cumulative'] = {t: C.evaluate(cs[:, t], goal, bins) for t in C.T_EVAL}

    # Action-free estimate (secondary).
    res['v_slope'] = {t: C.evaluate(C.v_slope_scores(q64, t), goal, bins) for t in C.T_EVAL}

    # Position baseline (reference).
    res['position_baseline'] = position_baseline(train_pos, train_goal, opos, goal, bins)

    # Moving frames only (diagnostic; true position + true goal).
    res['moving_only'] = moving_only(*m3, use & (dist_goal > C.NEAR_DIST), goal, bins,
                                     'distance to TRUE goal > 5 (as instructed)')
    res['moving_only_label_free'] = moving_only(*m3, use & (dist_all.min(axis=-1) > C.NEAR_DIST), goal, bins,
                                                'distance to ALL 3 candidate landmarks > 5')

    # Per-frame margin near vs far (diagnostic).
    near = dist_goal < C.NEAR_DIST
    res['margin'] = {}
    for beta in C.BETA_MAIN:
        ll = C.frame_loglik(*m3, beta)                                          # (N, T, 3)
        corr = ll[np.arange(N)[:, None], np.arange(T)[None], goal[:, None]]
        others = ll.copy()
        others[np.arange(N)[:, None], np.arange(T)[None], goal[:, None]] = -np.inf
        margin = corr - others.max(axis=-1)
        res['margin'][str(beta)] = dict(near_lt5=C.summarize(margin[near & use]),
                                        far_ge5=C.summarize(margin[~near & use]))

    # Critic action-sensitivity diagnostic: Delta Q = max_k - min_k over 64 directions.
    dq = q64.max(axis=-1) - q64.min(axis=-1)                                    # (N, T, 4)
    dq_true = dq[np.arange(N)[:, None], np.arange(T)[None], goal[:, None]]
    res['delta_q'] = {'true_r': {}, 'all_candidates': {}}
    for g in range(C.N_MAIN):
        m = goal == g
        res['delta_q']['true_r'][C.CAND_NAMES[g]] = dict(
            all=C.summarize(dq_true[m]), near_lt5=C.summarize(dq_true[m][near[m]]),
            far_ge5=C.summarize(dq_true[m][~near[m]]))
        res['delta_q']['all_candidates'][C.CAND_NAMES[g]] = {
            C.CAND_NAMES[j]: C.summarize(dq[m][..., j]) for j in range(len(C.CAND_R))}

    # Gate (§2.2) and interpretation rule (§2.7).
    a50 = res['irl_main'][50]['acc']
    gate = dict(irl_acc_t50=a50, colour_acc_t50=res['colour_cumulative'][50]['acc'],
                irl_acc_t20=res['irl_main'][20]['acc'], moving_only_acc=res['moving_only']['acc'])
    gate['pass'] = bool(a50 >= 0.8 and a50 > gate['colour_acc_t50'])
    res['gate'] = gate
    print('GATE:', json.dumps(gate))

    # Mean posterior curves for the figure.
    res['mean_post_correct_curve'] = [float(post_main[np.arange(N), t, goal].mean()) for t in range(T + 1)]

    out_path = os.path.join(SAVE_DIR, f'stage1_results{args.suffix}.json')
    with open(out_path, 'w') as fp:
        json.dump(res, fp, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f'Saved: {out_path}')

    for name in ('irl_main', 'irl_beta_ext', 'irl_4cand_with_blue', 'self_projection_prior',
                 'colour_cumulative', 'v_slope', 'position_baseline'):
        print(f'{name:28s} ' + '  '.join(f't={t}:{res[name][t]["acc"]:.3f}' for t in C.T_EVAL))
    print(f'{"self_projection_always_r1":28s} acc={res["self_projection_always_r1"]["acc"]:.3f}')
    for k in ('moving_only', 'moving_only_label_free'):
        print(f'{k:28s} acc={res[k]["acc"]:.3f} k_mean={res[k]["k_mean"]:.1f} by_k={res[k]["by_k"]}')


if __name__ == '__main__':
    main()
