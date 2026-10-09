"""
Extra checks on v7 stage 1 (docs/v7_experiment_log.md §2.8 items 4' and 5').
Neither is used for any gate.

4'. Is the action-free V-slope estimate's pull toward Red (= A-1's r1) an
    artefact of Q's scale differing between r? Standardize V(s, r) per r
    with the mean/sd over all TRAIN-split frames (A-2's true vision, as in
    stage 1), then redo the slope estimate on test. (Subtracting the mean
    does not change a slope; only the division by sd matters.)
5'. t=1 Red/Cyan confusion: confusion matrix at t=1 split by the sign of
    the first action's y component, and accuracy by first-action angle.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/stage1_extra_checks_v7.py --git_commit <hash>
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

DATA_PATH = os.path.join(C.PI_ROOT, 'data', 'data', 'v7_a1random_a2goal3', 'data.h5')
S1_DIR = os.path.join(C.PI_ROOT, 'data', 'result', 'v7_irl', 'stage1')


def train_v_stats(critic, f, device, batch_eps=200):
    """Mean and sd of V(s, r) = max_k Q(s, a_k, r) over all train frames, per candidate r."""
    ov = f['train/other_vision']
    n_ep, T = ov.shape[:2]
    s1 = np.zeros(len(C.CAND_R))
    s2 = np.zeros(len(C.CAND_R))
    cnt = 0
    for e0 in range(0, n_ep, batch_eps):
        v = ov[e0:e0 + batch_eps]
        m = v.shape[0] * T
        flat = v.reshape(m, *v.shape[2:])
        dummy = np.tile(C.PROBE_DIRS[:1], (m, 1))
        q64, _ = C.compute_q(critic, lambda i0, i1: flat[i0:i1].astype(np.float32) / 255.0, m, dummy, device)
        V = q64.max(-1).astype(np.float64)  # (m, R)
        s1 += V.sum(0)
        s2 += (V ** 2).sum(0)
        cnt += m
    mean = s1 / cnt
    sd = np.sqrt(s2 / cnt - mean ** 2)
    return mean, sd, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--git_commit', required=True)
    args = ap.parse_args()
    C.set_determinism(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f = h5py.File(DATA_PATH, 'r')
    goal = f['test/goal_index'][()]
    om = f['test/other_motion'][()]
    opos = f['test/other_position'][()]
    bins = C.init_bins(np.linalg.norm(opos[:, 0] - C.GOAL_POS[goal], axis=-1))
    q = np.load(os.path.join(S1_DIR, 'q_stage1.npz'))
    q64, qobs = q['q64'], q['qobs']
    out = dict(git_commit=args.git_commit, dataset='v7_a1random_a2goal3',
               true_values_used='A-2 true vision (train split for V statistics, test for the estimate), '
                                'A-2 true action (first action direction), true goal label (evaluation)')

    # 4'. standardized V slope
    critic = C.load_critic(device)
    mean, sd, cnt = train_v_stats(critic, f, device)
    out['v_train_stats'] = dict(n_frames=int(cnt), mean={C.CAND_NAMES[j]: float(mean[j]) for j in range(4)},
                                sd={C.CAND_NAMES[j]: float(sd[j]) for j in range(4)})
    q64_std = (q64.astype(np.float64) - mean[None, None, :, None]) / sd[None, None, :, None]
    out['v_slope_raw'] = {t: C.evaluate(C.v_slope_scores(q64, t), goal, bins) for t in C.T_EVAL}
    out['v_slope_standardized'] = {t: C.evaluate(C.v_slope_scores(q64_std, t), goal, bins) for t in C.T_EVAL}
    print('V train mean', out['v_train_stats']['mean'], 'sd', out['v_train_stats']['sd'])
    for k in ('v_slope_raw', 'v_slope_standardized'):
        print(k, {t: round(out[k][t]['acc'], 3) for t in C.T_EVAL},
              'pred-Red share t50:', round(np.array(out[k][50]['confusion'])[:, 0].sum() / len(goal), 3),
              'confusion t50:', out[k][50]['confusion'])

    # 5'. t=1 confusion by first-action y sign / angle
    lp1 = C.log_posterior_all_t(q64[:, :1, :3], qobs[:, :1, :3], np.ones((len(goal), 1), bool),
                                C.BETA_MAIN, np.log(np.full(3, 1 / 3)))[:, 1]
    pred, _ = C.argmax_with_ties(lp1)
    a0 = om[:, 0]
    ysign = np.where(a0[:, 1] >= 0, '+', '-')
    out['t1_by_first_action_y_sign'] = {}
    for s in ('+', '-'):
        m = ysign == s
        conf = np.zeros((3, 3), int)
        for g, p in zip(goal[m], pred[m]):
            conf[g, p] += 1
        out['t1_by_first_action_y_sign'][s] = dict(n=int(m.sum()), acc=float((pred[m] == goal[m]).mean()),
                                                   confusion=conf.tolist())
        print(f'y{s}: n={m.sum()} acc={(pred[m] == goal[m]).mean():.3f} confusion(rows true R/G/C)={conf.tolist()}')
    ang = (np.degrees(np.arctan2(a0[:, 1], a0[:, 0])) + 360) % 360
    abin = (((ang + 22.5) % 360) // 45).astype(int)  # 0 = east, 2 = north, 4 = west, 6 = south
    names = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    out['t1_by_first_action_angle'] = {}
    for b in range(8):
        e = {}
        for g in range(3):
            m = (abin == b) & (goal == g)
            e[C.CAND_NAMES[g]] = dict(n=int(m.sum()), acc=float((pred[m] == g).mean()) if m.any() else None,
                                      pred_counts=np.bincount(pred[m], minlength=3).tolist())
        out['t1_by_first_action_angle'][names[b]] = e
    for g in range(3):
        print(C.CAND_NAMES[g], {names[b]: (out['t1_by_first_action_angle'][names[b]][C.CAND_NAMES[g]]['n'],
                                           out['t1_by_first_action_angle'][names[b]][C.CAND_NAMES[g]]['pred_counts'])
                                for b in range(8)})
    path = os.path.join(S1_DIR, 'stage1_extra_checks.json')
    with open(path, 'w') as fp:
        json.dump(out, fp, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('Saved:', path)


if __name__ == '__main__':
    main()
