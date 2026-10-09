"""
v7 stage 2: state = viewpoint v̂² extracted from A-1's own vision
(Encoder-2 -> Decoder, see extract_vhat2_v7.py), action = A-2's TRUE action
(docs/v7_experiment_log.md §2.3, §2.8).

True values used in this stage:
  - inference: A-2's true action (other_motion) only. The state is v̂²,
    computed from A-1's self_vision only.
  - Q-agreement diagnostics: A-2's true vision (stage-1 Q, colour fractions)
  - diagnostics (moving frames, near/far): A-2's true position + true goal
  - baseline: A-2's true position (position classifier, train -> test)
  - evaluation: the true goal label

--cond true reruns the whole analysis on A-2's true vision (= stage 1) and
checks it reproduces stage1_results.json, so the shared analysis code here
is verified against the committed stage-1 numbers before it is used on v̂².

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/irl_stage2_v7.py --cond true --git_commit <hash>
    python analyze/irl_stage2_v7.py --cond a --git_commit <hash> [--suffix _run2]
    python analyze/irl_stage2_v7.py --cond b --git_commit <hash>
"""
import argparse
import json
import math
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, '/work/my_research/preference_inference/analyze')
import irl_common_v7 as C  # noqa
from irl_stage1_v7 import position_baseline, irl_by_t, moving_only  # noqa

DATASET = 'v7_a1random_a2goal3'
DATA_PATH = os.path.join(C.PI_ROOT, 'data', 'data', DATASET, 'data.h5')
S1_DIR = os.path.join(C.PI_ROOT, 'data', 'result', 'v7_irl', 'stage1')
SAVE_DIR = os.path.join(C.PI_ROOT, 'data', 'result', 'v7_irl', 'stage2')


def wilson_lower(p, n, z=1.959964):
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    return (centre - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def per_frame_margin(q64, qobs, goal, beta):
    ll = C.frame_loglik(q64, qobs, beta)  # (N, T, 3)
    T = ll.shape[1]
    idx = goal[:, None, None].repeat(T, 1)
    corr = np.take_along_axis(ll, idx, axis=-1)[..., 0]
    oth = ll.copy()
    np.put_along_axis(oth, idx, -np.inf, axis=-1)
    return corr - oth.max(-1)


def analyze_all(q64, qobs, frac, no_pix, d):
    """The stage-1 analysis (irl_stage1_v7.main), on any (Q, colour) pair."""
    N, T = qobs.shape[:2]
    goal, bins, use, near = d['goal'], d['bins'], d['use'], d['near']
    uniform3 = np.log(np.full(C.N_MAIN, 1 / C.N_MAIN))
    m3 = (q64[:, :, :C.N_MAIN], qobs[:, :, :C.N_MAIN])
    res = {}
    res['irl_main'], post_main = irl_by_t(*m3, use, C.BETA_MAIN, uniform3, goal, bins)
    res['irl_beta_ext'], _ = irl_by_t(*m3, use, C.BETA_EXT, uniform3, goal, bins)
    res['irl_4cand_with_blue'], _ = irl_by_t(q64, qobs, use, C.BETA_MAIN, np.log(np.full(4, 0.25)), goal, bins)
    res['self_projection_prior'], _ = irl_by_t(*m3, use, C.BETA_MAIN, np.log(np.array([0.5, 0.25, 0.25])),
                                               goal, bins)
    res['self_projection_always_r1'] = C.evaluate(np.tile(np.eye(C.N_MAIN)[C.R1_INDEX], (N, 1)), goal, bins)
    res['frac_frames_no_colour_pixel'] = float(no_pix.mean())
    cs = C.color_scores(frac)
    res['colour_cumulative'] = {t: C.evaluate(cs[:, t], goal, bins) for t in C.T_EVAL}
    res['v_slope'] = {t: C.evaluate(C.v_slope_scores(q64, t), goal, bins) for t in C.T_EVAL}
    res['position_baseline'] = position_baseline(d['train_pos'], d['train_goal'], d['opos'], goal, bins)
    res['moving_only'] = moving_only(*m3, use & (d['dist_goal'] > C.NEAR_DIST), goal, bins,
                                     'distance to TRUE goal > 5 (as instructed)')
    res['moving_only_label_free'] = moving_only(*m3, use & (d['dist_all'].min(axis=-1) > C.NEAR_DIST), goal, bins,
                                                'distance to ALL 3 candidate landmarks > 5')
    res['margin'] = {}
    for beta in C.BETA_MAIN:
        margin = per_frame_margin(*m3, goal, beta)
        res['margin'][str(beta)] = dict(near_lt5=C.summarize(margin[near & use]),
                                        far_ge5=C.summarize(margin[~near & use]))
    dq = q64.max(axis=-1) - q64.min(axis=-1)
    dq_true = dq[np.arange(N)[:, None], np.arange(T)[None], goal[:, None]]
    res['delta_q'] = {'true_r': {}, 'all_candidates': {}}
    for g in range(C.N_MAIN):
        m = goal == g
        res['delta_q']['true_r'][C.CAND_NAMES[g]] = dict(
            all=C.summarize(dq_true[m]), near_lt5=C.summarize(dq_true[m][near[m]]),
            far_ge5=C.summarize(dq_true[m][~near[m]]))
        res['delta_q']['all_candidates'][C.CAND_NAMES[g]] = {
            C.CAND_NAMES[j]: C.summarize(dq[m][..., j]) for j in range(len(C.CAND_R))}
    res['mean_post_correct_curve'] = [float(post_main[np.arange(N), t, goal].mean()) for t in range(T + 1)]
    return res


def q_agreement(q_true, q_hat, qobs_true, qobs_hat, frac_true, frac_hat, d):
    """§2.8 diagnostics 1-4: true A-2 vision vs v̂², under the same critic."""
    goal, near = d['goal'], d['near']
    out = {'margin_sign_agreement': {}, 'q_corr': {}, 'best_dir': {}, 'colour_corr': {}}
    mt = (q_true[:, :, :C.N_MAIN], qobs_true[:, :, :C.N_MAIN])
    mh = (q_hat[:, :, :C.N_MAIN], qobs_hat[:, :, :C.N_MAIN])
    for beta in C.BETA_MAIN:
        st = np.sign(per_frame_margin(*mt, goal, beta))
        sh = np.sign(per_frame_margin(*mh, goal, beta))
        agree = st == sh
        e = dict(all=float(agree.mean()),
                 frac_pos_true=float((st > 0).mean()), frac_pos_hat=float((sh > 0).mean()),
                 near_lt5=float(agree[near].mean()), far_ge5=float(agree[~near].mean()), by_goal={})
        for g in range(C.N_MAIN):
            m = goal == g
            e['by_goal'][C.CAND_NAMES[g]] = dict(
                all=float(agree[m].mean()), near_lt5=float(agree[m][near[m]].mean()),
                far_ge5=float(agree[m][~near[m]].mean()),
                frac_pos_hat=float((sh[m] > 0).mean()), frac_pos_true=float((st[m] > 0).mean()))
        out['margin_sign_agreement'][str(beta)] = e
    for g in range(C.N_MAIN):
        m = goal == g
        a, b = q_true[m][:, :, g], q_hat[m][:, :, g]                     # (n, T, 64)
        ac, bc = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
        ka, kb = a.argmax(-1), b.argmax(-1)
        ang = np.abs(((ka - kb) * 360.0 / C.K_DIRS + 180) % 360 - 180)
        out['q_corr'][C.CAND_NAMES[g]] = dict(
            raw=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
            action_centred=float(np.corrcoef(ac.ravel(), bc.ravel())[0, 1]),
            near_action_centred=float(np.corrcoef(ac[near[m]].ravel(), bc[near[m]].ravel())[0, 1]),
            far_action_centred=float(np.corrcoef(ac[~near[m]].ravel(), bc[~near[m]].ravel())[0, 1]))
        out['best_dir'][C.CAND_NAMES[g]] = dict(
            exact_match=float((ka == kb).mean()), mean_abs_angle_deg=float(ang.mean()),
            near_mean_abs_angle_deg=float(ang[near[m]].mean()), far_mean_abs_angle_deg=float(ang[~near[m]].mean()))
    for j, name in enumerate(('red', 'green', 'blue', 'cyan')):
        x, y = frac_true[..., j].ravel(), frac_hat[..., j].ravel()
        out['colour_corr'][name] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', required=True, choices=['true', 'a', 'b'])
    ap.add_argument('--git_commit', required=True)
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()

    C.set_determinism(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f = h5py.File(DATA_PATH, 'r')
    ov = f['test/other_vision'][()]
    om = f['test/other_motion'][()]
    opos = f['test/other_position'][()]
    goal = f['test/goal_index'][()]
    N, T = om.shape[:2]
    dist_goal = np.linalg.norm(opos - C.GOAL_POS[goal][:, None], axis=-1)
    d = dict(goal=goal, opos=opos, dist_goal=dist_goal,
             dist_all=np.linalg.norm(opos[:, :, None] - C.GOAL_POS[None, None], axis=-1),
             bins=C.init_bins(dist_goal[:, 0]), near=dist_goal < C.NEAR_DIST,
             train_pos=f['train/other_position'][()], train_goal=f['train/goal_index'][()])
    unit, small = C.unit_actions(om)
    d['use'] = ~small

    if args.cond == 'true':
        vis = ov.reshape(N * T, *ov.shape[2:]).astype(np.float32) / 255.0
    else:
        vis = h5py.File(os.path.join(SAVE_DIR, f'vhat2_{args.cond}_test.h5'), 'r')['vhat2'][()]
        vis = vis.reshape(N * T, *vis.shape[2:]).astype(np.float32)

    critic = C.load_critic(device)
    q64, qobs = C.compute_q(critic, lambda i0, i1: vis[i0:i1], N * T, unit.reshape(N * T, 2), device)
    q64 = q64.reshape(N, T, len(C.CAND_R), C.K_DIRS)
    qobs = qobs.reshape(N, T, len(C.CAND_R))
    frac, no_pix = C.landmark_fraction_array(vis)
    frac = frac.reshape(N, T, 4)

    os.makedirs(SAVE_DIR, exist_ok=True)
    res = analyze_all(q64, qobs, frac, no_pix, d)

    if args.cond == 'true':
        s1 = json.load(open(os.path.join(S1_DIR, 'stage1_results.json')))
        mine = json.loads(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
        same = {k: mine[k] == s1[k] for k in mine}
        print('reproduces stage1_results.json:', all(same.values()), {k: v for k, v in same.items() if not v})
        if not all(same.values()):
            sys.exit(1)
        return

    np.savez_compressed(os.path.join(SAVE_DIR, f'q_stage2_{args.cond}{args.suffix}.npz'), q64=q64, qobs=qobs)
    s1q = np.load(os.path.join(S1_DIR, 'q_stage1.npz'))
    frac_true, no_pix_true = C.landmark_fraction_array(ov.reshape(N * T, *ov.shape[2:]).astype(np.float32) / 255.0)
    agreement = q_agreement(s1q['q64'], q64, s1q['qobs'], qobs, frac_true.reshape(N, T, 4), frac, d)
    agreement['frac_frames_no_colour_pixel'] = dict(true_vision=float(no_pix_true.mean()), vhat2=float(no_pix.mean()))

    s1 = json.load(open(os.path.join(S1_DIR, 'stage1_results.json')))
    a50 = res['irl_main'][50]['acc']
    gate = dict(irl_acc_t50=a50, wilson95_lower=wilson_lower(a50, N), chance=1 / 3,
                always_r1_acc=res['self_projection_always_r1']['acc'],
                irl_acc_t20=res['irl_main'][20]['acc'], moving_only_acc=res['moving_only']['acc'])
    gate['pass'] = bool(gate['wilson95_lower'] > 1 / 3 and gate['wilson95_lower'] > gate['always_r1_acc'])
    out = dict(
        experiment=f'v7_s2_vhat2_{args.cond}_true_action', git_commit=args.git_commit, cond=args.cond,
        vhat2_meta=json.load(open(os.path.join(SAVE_DIR, f'vhat2_{args.cond}_meta.json'))),
        dataset=DATASET, split='test', critic=C.CRITIC_PATH, seeds=dict(torch=0, numpy=0, tie_break=0),
        true_values_used=dict(
            inference='A-2 true action (other_motion) only; state = v̂² from A-1 self_vision',
            q_agreement='A-2 true vision (stage-1 Q and colour fractions)',
            diagnostics='A-2 true position and true goal label (moving-frame selection, near/far split)',
            baseline='A-2 true position (position classifier, train split -> test split)',
            evaluation='true goal label'),
        n_test=int(N), frac_small_action=float(small.mean()),
        q_agreement=agreement, gate=gate,
        diff_from_stage1={name: {t: res[name][t]['acc'] - s1[name][str(t)]['acc'] for t in C.T_EVAL}
                          for name in ('irl_main', 'colour_cumulative', 'v_slope')},
        **res)
    print('GATE:', json.dumps(gate))
    path = os.path.join(SAVE_DIR, f'stage2_{args.cond}_results{args.suffix}.json')
    with open(path, 'w') as fp:
        json.dump(out, fp, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f'Saved: {path}')
    for name in ('irl_main', 'irl_beta_ext', 'irl_4cand_with_blue', 'self_projection_prior',
                 'colour_cumulative', 'v_slope', 'position_baseline'):
        print(f'{name:28s} ' + '  '.join(f't={t}:{res[name][t]["acc"]:.3f}' for t in C.T_EVAL))
    for k in ('moving_only', 'moving_only_label_free'):
        print(f'{k:28s} acc={res[k]["acc"]:.3f}')
    print('margin sign agreement:', {b: round(v['all'], 3) for b, v in agreement['margin_sign_agreement'].items()})
    print('q_corr:', agreement['q_corr'])
    print('best_dir:', agreement['best_dir'])
    print('colour:', agreement['colour_corr'], agreement['frac_frames_no_colour_pixel'])
    print('confusion t50:', res['irl_main'][50]['confusion'], 'mean_post_r1 t50:', res['irl_main'][50]['mean_post_r1'])


if __name__ == '__main__':
    main()
