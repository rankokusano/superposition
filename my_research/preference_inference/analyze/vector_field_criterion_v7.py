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

Mixed goals (remake 1, §2.13): same idea with the target direction being the
direction to the NEARER of the two corners, on grid points whose split
coordinate satisfies |x| (or |y|) >= 3:
  old criterion  : near-corner agreement >= 0.7 (read from feasibility_results.json)
  new criterion  : critic_angle <= constant_flow - 15 deg
  check2 pass    : both

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
CORNER = {'red': np.array([-9.0, 9.0]), 'green': np.array([-9.0, -9.0]),
          'blue': np.array([9.0, -9.0]), 'cyan': np.array([9.0, 9.0])}
# mixed goal -> (corner if split coord < 0, corner if >= 0, split axis) -- as feasibility_goals_v7.MIXED
MIXED = {'Mtop': ('red', 'cyan', 0), 'Mleft': ('green', 'red', 1),
         'Mbottom': ('green', 'blue', 0), 'Mright': ('blue', 'cyan', 1)}
OUT = os.environ.get('VF_OUT', os.path.join(PI_ROOT, 'data', 'result', 'v7_irl', 'retrain', 'vector_field_criterion.json'))


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


def judge(maps, feas=None):
    xy = maps['xy']
    out = {}
    for goal, (neg, posc, ax) in MIXED.items():
        if goal not in maps.files:
            continue
        m = np.abs(xy[:, ax]) >= 3
        near = np.where(xy[:, ax] < 0, 0, 1)
        corners = np.stack([CORNER[neg], CORNER[posc]])
        to_near = ang_deg(corners[near] - xy)[m]
        mu = circ_mean_deg(to_near)
        const = float(ang_diff(mu, to_near).mean())
        crit = float(ang_diff(maps[goal][m], to_near).mean())
        agree = None if feas is None else feas['check2'][goal]['near_corner_agreement']
        old_ok = None if agree is None else bool(agree >= 0.7)
        new_ok = bool(crit <= const - 15.0)
        out[goal] = dict(near_corner_agreement=agree, critic_angle_to_near_deg=crit, constant_flow_deg=const,
                         constant_flow_direction_deg=mu, circ_var=circ_var(maps[goal][m]),
                         old_criterion_pass=old_ok, new_criterion_pass=new_ok,
                         check2_pass=bool(old_ok and new_ok) if old_ok is not None else None)
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
        fpath = os.path.join(PI_ROOT, d, 'feasibility_results.json')
        feas = json.load(open(fpath)) if os.path.exists(fpath) else None
        res[label] = judge(np.load(os.path.join(PI_ROOT, d, 'check2_maps.npz')), feas)
        for goal, e in res[label].items():
            if goal in MIXED:
                print(f'{label:10s} {goal:7s} agree {e["near_corner_agreement"]}  critic {e["critic_angle_to_near_deg"]:6.1f}  '
                      f'const-flow {e["constant_flow_deg"]:6.1f}  circvar {e["circ_var"]:.3f}  old={e["old_criterion_pass"]} '
                      f'new={e["new_criterion_pass"]} check2={e["check2_pass"]}')
            else:
                print(f'{label:10s} {goal:7s} critic {e["critic_angle_deg"]:6.1f}  const-flow {e["constant_flow_deg"]:6.1f} '
                      f'(dir {e["constant_flow_direction_deg"]:5.1f})  circvar {e["circ_var"]:.3f}  '
                      f'old={e["old_criterion_pass"]} new={e["new_criterion_pass"]} vector_field={e["vector_field"]}')
    with open(OUT, 'w') as fp:
        json.dump(res, fp, indent=1)
    print('Saved:', OUT)


if __name__ == '__main__':
    main()
