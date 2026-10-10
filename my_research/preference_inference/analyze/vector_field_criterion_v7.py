"""
Single-goal "vector field vs constant flow" judgement for retrained critics
(docs/v7_experiment_log.md §2.11 criterion 2-3, §2.12). Reads the
check2_maps.npz written by feasibility_goals_v7.py --check2.

For each single goal (Red, Green, Cyan) on the 20x20 grid:
  critic_angle   = mean_i |best_dir_i - dir_to_goal_i|
  constant_flow  = mean_i |mu - dir_to_goal_i|, mu = circular mean of dir_to_goal_i
                   (the best a single fixed direction does, approximately)
  old criterion  : critic_angle <= 30 deg
  new criterion  : critic_angle <= constant_flow - 15 deg
  vector field   : both criteria pass (§2.12)
Circular variance of best_dir is reported alongside.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/vector_field_criterion_v7.py v6_seed2=data/result/v7_irl/retrain/v6_seed2_ref \
        A_seed2=data/result/v7_irl/retrain/seed2_condA ...
"""
import json
import os
import sys

import numpy as np

PI_ROOT = '/work/my_research/preference_inference'
LM = {'Red': np.array([-9.0, 9.0]), 'Green': np.array([-9.0, -9.0]), 'Cyan': np.array([9.0, 9.0])}
OUT = os.path.join(PI_ROOT, 'data', 'result', 'v7_irl', 'retrain', 'vector_field_criterion.json')


def ang_deg(v):
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360


def ang_diff(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def circ_mean_deg(deg):
    rad = np.radians(deg)
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


def circ_var(deg):
    rad = np.radians(deg)
    return float(1 - np.hypot(np.cos(rad).mean(), np.sin(rad).mean()))


def judge(maps):
    xy = maps['xy']
    out = {}
    for goal, p in LM.items():
        to_goal = ang_deg(p - xy)
        mu = circ_mean_deg(to_goal)
        const = float(ang_diff(mu, to_goal).mean())
        crit = float(ang_diff(maps[goal], to_goal).mean())
        old_ok = crit <= 30.0
        new_ok = crit <= const - 15.0
        out[goal] = dict(critic_angle_deg=crit, constant_flow_deg=const, constant_flow_direction_deg=mu,
                         circ_var=circ_var(maps[goal]), old_criterion_pass=bool(old_ok),
                         new_criterion_pass=bool(new_ok), vector_field=bool(old_ok and new_ok))
    return out


def main():
    res = {}
    for pair in sys.argv[1:]:
        label, d = pair.split('=', 1)
        res[label] = judge(np.load(os.path.join(PI_ROOT, d, 'check2_maps.npz')))
        for goal, e in res[label].items():
            print(f'{label:10s} {goal:5s} critic {e["critic_angle_deg"]:6.1f}  const-flow {e["constant_flow_deg"]:6.1f} '
                  f'(dir {e["constant_flow_direction_deg"]:5.1f})  circvar {e["circ_var"]:.3f}  '
                  f'old={e["old_criterion_pass"]} new={e["new_criterion_pass"]} vector_field={e["vector_field"]}')
    with open(OUT, 'w') as fp:
        json.dump(res, fp, indent=1)
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
