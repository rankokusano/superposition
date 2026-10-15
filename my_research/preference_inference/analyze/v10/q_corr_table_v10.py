"""
v10 (user request 2026-10-16, item 6): correlation between the two layouts'
critic value maps for every B1 seed (A-1 / A-2, seeds 0-5) and the v3
critics, next to each seed's gate B1 result; plus the reference value: the
correlation between the two layouts' TRUE reward maps (A-1: red - green from
its own camera; A-2: green - red from its own camera), rendered with the same
placement as the value maps (21x21 grid, the other agent at (0, 0)). This is
the value a critic would reach if its value followed the colour completely.

Value-map correlations reported:
  q8   : Q(s, a_probe), 441 points x 8 probes (as reported in B1)
  qmax : max over the 8 probes, 441 points (the Q-value map in the figures)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR, git_hash  # noqa
sys.path.insert(0, PI_DIR)
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

from sim_v10 import creator, layout  # noqa
from sim_v10.util import load_config  # noqa

GRID = np.arange(-10, 11, dtype=float)


def reward_maps(camera, sign):
    env = creator.create_environment(load_config(os.path.join(
        PI_DIR, 'sim_v10', 'config', 'collect', 'self_random_other_stay.yml')).environment)
    env.init()
    env.off_display()
    env.reset()
    out = {}
    for lo in layout.LAYOUT_NAMES:
        env.set_layout(lo)
        m = np.zeros((21, 21))
        for i, x in enumerate(GRID):
            for j, y in enumerate(GRID):
                if camera == 'self':
                    env.self_agent.p, env.other_agent.p = np.array([x, y]), np.array([0.0, 0.0])
                else:
                    env.self_agent.p, env.other_agent.p = np.array([0.0, 0.0]), np.array([x, y])
                env.world.set_camera(camera)
                env.world.draw(env.self_agent, env.other_agent)
                f = layout.color_fractions(env.world.capture())
                m[i, j] = sign * (f['Red'] - f['Green'])
        out[lo] = m
    return out


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    res = {'git_commit': git_hash(), 'placement': '21x21 grid, other agent at (0,0)', 'reward_reference': {}, 'critics': []}
    for ag, cam, sign in (('a1', 'self', 1.0), ('a2', 'other', -1.0)):
        rm = reward_maps(cam, sign)
        res['reward_reference'][ag] = {'corr_reward_original_vs_swap': corr(rm['original'], rm['swap_rg']),
                                       'reward_range': [float(min(m.min() for m in rm.values())),
                                                        float(max(m.max() for m in rm.values()))]}
    v3 = np.load(os.path.join(OUT_ROOT, 'stageA', 'direction_maps', 'direction_maps.npz'))
    for ag in ('a1', 'a2'):
        A = ag.upper()
        res['critics'].append({'agent': ag, 'critic': 'v3 (stage A)', 'gate': 'n/a (trained in the original layout only)',
                               'corr_q8': corr(v3[f'{A}_original_q8'], v3[f'{A}_swap_rg_q8']),
                               'corr_qmax': corr(v3[f'{A}_original_qmax8'], v3[f'{A}_swap_rg_qmax8'])})
        for s in range(6):
            d = os.path.join(OUT_ROOT, 'b1', f'{ag}_seed{s}')
            arr = np.load(os.path.join(d, 'b1_arrays.npz'))
            g = json.load(open(os.path.join(d, 'b1_checks.json')))
            res['critics'].append({
                'agent': ag, 'critic': f'B1 seed{s}',
                'gate': {'check1_health': g['check1_health']['pass'], 'check2_rollout': g['check2_rollout']['pass'],
                         'check3_direction': g['check3_direction']['pass'], 'pass_all': g['pass_all']},
                'action_std': [g['check1_health'][lo]['action_std'] for lo in layout.LAYOUT_NAMES],
                'state_std': [g['check1_health'][lo]['state_std'] for lo in layout.LAYOUT_NAMES],
                'corr_q8': corr(arr['original_q8'], arr['swap_rg_q8']),
                'corr_qmax': corr(arr['original_qmax8'], arr['swap_rg_qmax8'])})
    out = os.path.join(OUT_ROOT, 'b1', 'q_corr_table.json')
    json.dump(res, open(out, 'w'), indent=2)
    print(json.dumps(res['reward_reference'], indent=1))
    for r in res['critics']:
        gate = r['gate'] if isinstance(r['gate'], str) else ' '.join(f'{k[:6]}={int(v)}' for k, v in r['gate'].items())
        print(f"{r['agent']} {r['critic']:14s} q8 {r['corr_q8']:+.3f} qmax {r['corr_qmax']:+.3f}  {gate}")


if __name__ == '__main__':
    main()
