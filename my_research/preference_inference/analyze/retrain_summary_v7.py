"""
Pre-registered judgement for a retraining round (docs/v7_experiment_log.md
§2.11-§2.13), from the files the pipeline writes. Prints and saves a summary;
makes no decision beyond the registered criteria.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/retrain_summary_v7.py --round r1 --critic_seed 3 --actor_seed 4 --b_seed 2
"""
import argparse
import json
import os
import re

PI = '/work/my_research/preference_inference'
R = os.path.join(PI, 'data', 'result', 'v7_irl', 'retrain')


def collapsed(health_txt):
    stds = [float(x) for x in re.findall(r'std across corners: ([0-9.]+)', open(health_txt).read())]
    return len(stds) == 3 and all(s < 0.001 for s in stds), stds


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', required=True)
    ap.add_argument('--critic_seed', type=int, required=True)
    ap.add_argument('--actor_seed', type=int, required=True)
    ap.add_argument('--b_seed', type=int, required=True)
    a = ap.parse_args()
    tagA, tagB = f'condA_{a.round}', f'condB_{a.round}'
    names = {'critic': f'seed{a.critic_seed}_{tagA}', 'actor': f'seed{a.actor_seed}_{tagA}', 'B': f'seed{a.b_seed}_{tagB}'}
    vf = load(os.path.join(R, f'vector_field_criterion_{a.round}.json')) or {}
    out = {'round': a.round, 'models': names}
    for role, name in names.items():
        h = os.path.join(R, f'health_{name}.txt')
        c, stds = collapsed(h) if os.path.exists(h) else (None, None)
        out[f'{role}_collapsed'] = c
        out[f'{role}_corner_std'] = stds
    fc = load(os.path.join(R, names['critic'], 'feasibility_results.json'))
    fa = load(os.path.join(R, names['actor'], 'feasibility_results.json'))
    vc = vf.get(names['critic'], {})
    req = ['Red', 'Cyan', 'Mtop']
    if fc and not out['critic_collapsed']:
        c1 = {g: fc['check1'][g]['pass'] for g in req}
        c2 = {'Red': True, 'Cyan': True, 'Mtop': bool(vc.get('Mtop', {}).get('check2_pass'))}
        out['A_check1_critic_seed'] = c1
        out['A_check2_critic'] = c2
        out['A_pass'] = all(c1.values()) and all(c2.values())
    else:
        out['A_pass'] = False
    if fa:
        out['A2_actor_check1'] = {g: fa['check1'][g]['pass'] for g in req}
        out['A2_actor_pass'] = all(out['A2_actor_check1'].values())
    out['vector_field'] = {k: {g: v[g]['vector_field'] for g in ('Red', 'Green', 'Cyan')} for k, v in vf.items()}
    if names['B'] in vf and 'v6_seed2' in vf and not out['B_collapsed']:
        b, v6 = vf[names['B']], vf['v6_seed2']
        old = all(b[g]['critic_angle_deg'] <= v6[g]['critic_angle_deg'] - 15 and b[g]['critic_angle_deg'] <= 30
                  for g in ('Red', 'Green', 'Cyan'))
        new = all(b[g]['new_criterion_pass'] for g in ('Red', 'Green', 'Cyan'))
        out['B_hypothesis'] = dict(old_criterion=old, new_criterion=new, supported=old and new)
    else:
        out['B_hypothesis'] = 'not testable (B collapsed or missing)'
    path = os.path.join(R, f'summary_{a.round}.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(json.dumps(out, indent=1))
    print('Saved:', path)


if __name__ == '__main__':
    main()
