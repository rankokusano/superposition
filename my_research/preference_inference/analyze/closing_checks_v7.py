"""
v7 closing checks (docs/v7_experiment_log.md §2.14), from rollouts written by
feasibility_goals_v7.py --rollout. No training; not used for any gate.

Check A: per goal, mean cumulative reward per episode
    sum_{t=0..99} reward_from_vision(view at t+1, r_goal)
  (the same reward the actor was trained on: next view after the other agent's
  random-walk step) and the share of episodes ending within 5 of each corner,
  for the trained actors and the fixed-direction policies run with the same
  rollout key (same initial positions and A-1 walks). Support rule: for Red,
  Green and Cyan, max(fixed critic-direction, fixed corner-direction) >= 0.9 x
  actor, for BOTH actor comparisons.
Check B: check1 metrics with the actor's hidden reset every step vs carried.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/closing_checks_v7.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/work')
from my_research.preference_inference.model.rl_agent_sac_v6 import reward_from_vision  # noqa

PI = '/work/my_research/preference_inference'
R = os.path.join(PI, 'data', 'result', 'v7_irl')
C = os.path.join(R, 'retrain', 'closing')
GOALS = ['Red', 'Green', 'Cyan', 'Mtop', 'Mleft', 'Mbottom', 'Mright']
SINGLE = ['Red', 'Green', 'Cyan']
LM = {'red': np.array([-9.0, 9.0]), 'green': np.array([-9.0, -9.0]),
      'blue': np.array([9.0, -9.0]), 'cyan': np.array([9.0, 9.0])}
MIXED = {'Mtop': ('red', 'cyan', 0), 'Mleft': ('green', 'red', 1),
         'Mbottom': ('green', 'blue', 0), 'Mright': ('blue', 'cyan', 1)}
SINGLE_LM = {'Red': 'red', 'Green': 'green', 'Cyan': 'cyan'}
COMPARISONS = {
    'A_seed3': dict(actor=os.path.join(R, 'retrain', 'seed3_condA_r1'),
                    fixed_critic=os.path.join(C, 'fixed_crit_seed3'),
                    fixed_corner=os.path.join(C, 'fixed_corner_seed3key')),
    'v6_seed1': dict(actor=os.path.join(R, 'feasibility'),
                     fixed_critic=os.path.join(C, 'fixed_crit_v6'),
                     fixed_corner=os.path.join(C, 'fixed_corner_v6key')),
}
HIDDEN = {'A_seed1': (os.path.join(R, 'retrain', 'seed1_condA'), os.path.join(C, 'hidreset_seed1')),
          'A_seed3': (os.path.join(R, 'retrain', 'seed3_condA_r1'), os.path.join(C, 'hidreset_seed3'))}


def rollout_stats(d, goal):
    z = np.load(os.path.join(d, f'rollout_{goal}.npz'))
    vis, pos, r = z['vis'], z['pos'], z['r']
    rew = np.array([[reward_from_vision(vis[n, t] / 255.0, r) for t in range(1, vis.shape[1])]
                    for n in range(vis.shape[0])])
    cum = rew.sum(1)
    pf = pos[:, -1]
    ends = {k: float((np.linalg.norm(pf - v, axis=1) <= 5).mean()) for k, v in LM.items()}
    out = dict(mean_cum_reward=float(cum.mean()), sd_cum_reward=float(cum.std()), end_within5=ends,
               start_pos_md5=str(hash(pos[:, 0].tobytes())))
    if 'fixed_deg' in z.files and not np.isnan(z['fixed_deg']):
        out['fixed_deg'] = float(z['fixed_deg'])
    return out, pos


def check1_metrics(d, goal):
    z = np.load(os.path.join(d, f'rollout_{goal}.npz'))
    p0, pf = z['pos'][:, 0], z['pos'][:, -1]
    if goal in SINGLE_LM:
        return dict(mean_final_dist=float(np.linalg.norm(pf - LM[SINGLE_LM[goal]], axis=1).mean()))
    neg, posc, ax = MIXED[goal]
    near = np.where(p0[:, ax] < 0, neg, posc)
    m = np.abs(p0[:, ax]) >= 2
    at_near = np.array([np.linalg.norm(pf[i] - LM[c]) <= 5 for i, c in enumerate(near)])
    return dict(frac_end_near=float(at_near[m].mean()),
                end_within5={k: float((np.linalg.norm(pf - v, axis=1) <= 5).mean()) for k, v in LM.items()})


def main():
    res = {'check_A': {}, 'check_B': {}}
    for comp, dirs in COMPARISONS.items():
        e = {}
        for goal in GOALS:
            row = {}
            starts = {}
            for k, d in dirs.items():
                row[k], pos = rollout_stats(d, goal)
                starts[k] = pos[:, 0]
            row['same_initial_positions'] = bool(all(np.array_equal(starts['actor'], v) for v in starts.values()))
            best = max(row['fixed_critic']['mean_cum_reward'], row['fixed_corner']['mean_cum_reward'])
            a = row['actor']['mean_cum_reward']
            row['best_fixed_over_actor'] = float(best / a) if a > 0 else None
            e[goal] = row
            print(f'[A] {comp:8s} {goal:7s} actor {a:7.2f}  fixed_critic {row["fixed_critic"]["mean_cum_reward"]:7.2f} '
                  f'({row["fixed_critic"].get("fixed_deg", float("nan")):5.1f}°)  fixed_corner '
                  f'{row["fixed_corner"]["mean_cum_reward"]:7.2f}  ratio {row["best_fixed_over_actor"]}  '
                  f'same_starts={row["same_initial_positions"]}')
        e['support'] = bool(all(e[g]['best_fixed_over_actor'] is not None and e[g]['best_fixed_over_actor'] >= 0.9
                                for g in SINGLE))
        res['check_A'][comp] = e
    res['check_A']['hypothesis_supported'] = bool(all(res['check_A'][c]['support'] for c in COMPARISONS))
    print('check A: hypothesis supported =', res['check_A']['hypothesis_supported'])
    for name, (carried, reset) in HIDDEN.items():
        res['check_B'][name] = {g: dict(hidden_carried=check1_metrics(carried, g), hidden_reset=check1_metrics(reset, g))
                                for g in GOALS}
        for g in GOALS:
            print(f'[B] {name} {g:7s} carried {res["check_B"][name][g]["hidden_carried"]}  reset {res["check_B"][name][g]["hidden_reset"]}')
    out = os.path.join(C, 'closing_checks.json')
    json.dump(res, open(out, 'w'), indent=1)
    print('Saved:', out)


if __name__ == '__main__':
    main()
