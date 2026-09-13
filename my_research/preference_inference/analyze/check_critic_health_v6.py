"""
v6 S1 critic health check: extends analyze/check_critic_health.py (which
stays untouched, per the "copy into preference_inference/, don't edit the
shared original" rule) to a reward-conditioned Q(s,a,r).

Checks required by docs/v6_instructions.md §3 (S1 gate) plus the
2026-09-13 addition in docs/v6_experiment_log.md §2 (generalization to an
unseen r):

  1. action sensitivity (per r): same vision, several probe actions
  2. state sensitivity (per r): 4 corners + center
  3. OOD sensitivity (per r): zeros/ones/random vision
  4. reward-condition response: r=A1 (+1,-1,0,0) and r=A2 (-1,+1,0,0) must
     give CLEARLY DIFFERENT spatial Q maps (A1's should peak at Red, A2's
     at Green)
  5. value-range match: std(Q) across corners must be the same order of
     magnitude for r=A1 vs r=A2 (this is the whole point of v6 -- v4/v5's
     A1/A2-specific critics had a 0.63 vs 0.009 std gap)
  6. generalization to an unseen r: r=(0,+1,-1,0) (never sampled during
     training under continuous-uniform sampling almost surely, but on the
     same L1=2 shell) should still peak at Green and trough at Blue

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/check_critic_health_v6.py \
        --critic_path data/model/v6_rl_critic_seed0.pth \
        --env_config /work/simulation/config/collect/self_random_other_stay.yml \
        --camera self --scan_self --label v6_seed0
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

import creator  # noqa
from util import load_config  # noqa
from my_research.preference_inference.model.rl_agent_sac_v6 import (  # noqa
    CriticLSTM, A1_TRUE_R, A2_TRUE_R,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 'Yellow' is the name used elsewhere in the codebase for historical
# reasons; it renders as pixel-pure cyan (0,1,1) -- confirmed empirically
# in docs/v6_experiment_log.md §4.1 and again here via landmark_fractions.
CORNERS = {
    'Red':    (-9.0, 9.0),
    'Green':  (-9.0, -9.0),
    'Blue':   (9.0, -9.0),
    'Cyan(Yellow)': (9.0, 9.0),
    'Center': (0.0, 0.0),
}

PROBE_ACTIONS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, -0.7)]

R_CONDITIONS = {
    'A1(+1,-1,0,0)': np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32),
    'A2(-1,+1,0,0)': np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    'unseen(0,+1,-1,0)': np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float32),
}


def vision_to_tensor(v):
    v = v.astype(np.float32)
    return torch.tensor(np.transpose(v, (2, 0, 1))).unsqueeze(0).to(DEVICE)


def r_to_tensor(w):
    return torch.tensor(np.asarray(w, dtype=np.float32)).unsqueeze(0).to(DEVICE)


def q_of(critic, v_t, action, r_t):
    a_t = torch.tensor([action], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        q, _ = critic(v_t, a_t, r_t, hidden=None)
    return q.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic_path', required=True)
    parser.add_argument('--env_config', required=True)
    parser.add_argument('--camera', choices=['self', 'other'], required=True)
    parser.add_argument('--scan_self', action='store_true')
    parser.add_argument('--label', default='critic_v6')
    parser.add_argument('--action-eps', type=float, default=0.05)
    parser.add_argument('--corner-eps', type=float, default=0.05)
    parser.add_argument('--range-ratio-eps', type=float, default=3.0,
                         help='max allowed ratio between the largest and smallest '
                              'per-r corner-std before flagging a value-range mismatch '
                              '(v4 saw 0.63/0.009 ~ 70x; this should be close to 1x)')
    args = parser.parse_args()

    env_cfg = load_config(args.env_config)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()
    env.reset()
    if args.scan_self:
        env.other_agent.p = np.array([0.0, 0.0])
    else:
        env.self_agent.p = np.array([0.0, 0.0])

    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(args.critic_path, map_location=DEVICE))
    critic.eval()

    def render_at(p):
        agent = env.self_agent if args.scan_self else env.other_agent
        agent.p = np.array(p, dtype=float)
        env.world.set_camera(args.camera)
        env.world.draw(env.self_agent, env.other_agent)
        return env.world.capture()

    print(f'=== v6 critic health check: {args.label} ({args.critic_path}) ===')

    corner_std_by_r = {}
    corner_qs_by_r = {}
    action_std_by_r = {}

    for r_name, r_vec in R_CONDITIONS.items():
        r_t = r_to_tensor(r_vec)
        print(f'\n--- r = {r_name} ---')

        # 1. action sensitivity
        v = render_at(CORNERS['Red'])
        v_t = vision_to_tensor(v)
        action_qs = [q_of(critic, v_t, a, r_t) for a in PROBE_ACTIONS]
        action_std = float(np.std(action_qs))
        action_std_by_r[r_name] = action_std
        print(f'  1) action sensitivity (vision@Red): std={action_std:.6f} '
              f'(threshold {args.action_eps})')

        # 2. state sensitivity (real corners)
        corner_qs = {}
        for name, p in CORNERS.items():
            v = render_at(p)
            v_t = vision_to_tensor(v)
            corner_qs[name] = q_of(critic, v_t, (1.0, 0.0), r_t)
        corner_qs_by_r[r_name] = corner_qs
        corner_std = float(np.std(list(corner_qs.values())))
        corner_std_by_r[r_name] = corner_std
        print(f'  2) state sensitivity (action=(1,0)): '
              + ', '.join(f'{n}={q:.4f}' for n, q in corner_qs.items()))
        print(f'     std across corners: {corner_std:.6f} (threshold {args.corner_eps})')
        argmax_corner = max(corner_qs, key=corner_qs.get)
        argmin_corner = min(corner_qs, key=corner_qs.get)
        print(f'     argmax={argmax_corner}  argmin={argmin_corner}')

        # 3. OOD sensitivity
        real_v = vision_to_tensor(render_at(CORNERS['Red']))
        ood_inputs = {
            'zeros': torch.zeros_like(real_v),
            'ones': torch.ones_like(real_v),
            'random': torch.rand_like(real_v),
        }
        ood_qs = {n: q_of(critic, v_t, (1.0, 0.0), r_t) for n, v_t in ood_inputs.items()}
        print(f'  3) OOD sensitivity: ' + ', '.join(f'{n}={q:.4f}' for n, q in ood_qs.items()))

    # 4. reward-condition response: A1 should peak at Red, A2 at Green
    a1_argmax = max(corner_qs_by_r['A1(+1,-1,0,0)'], key=corner_qs_by_r['A1(+1,-1,0,0)'].get)
    a2_argmax = max(corner_qs_by_r['A2(-1,+1,0,0)'], key=corner_qs_by_r['A2(-1,+1,0,0)'].get)
    print(f'\n4) reward-condition response:')
    print(f'   A1(+1,-1,0,0) peaks at: {a1_argmax}  (expect Red)')
    print(f'   A2(-1,+1,0,0) peaks at: {a2_argmax}  (expect Green)')
    response_ok = (a1_argmax == 'Red') and (a2_argmax == 'Green')

    # 5. value-range match across r conditions
    stds = [corner_std_by_r[n] for n in ('A1(+1,-1,0,0)', 'A2(-1,+1,0,0)')]
    range_ratio = max(stds) / max(min(stds), 1e-8)
    print(f'\n5) value-range match: corner-std A1={stds[0]:.6f} A2={stds[1]:.6f} '
          f'ratio={range_ratio:.2f}x (threshold {args.range_ratio_eps}x; '
          f'v4 had a 0.63/0.009 ~ 70x gap)')
    range_ok = range_ratio <= args.range_ratio_eps

    # 6. generalization to an unseen r
    unseen_argmax = max(corner_qs_by_r['unseen(0,+1,-1,0)'], key=corner_qs_by_r['unseen(0,+1,-1,0)'].get)
    unseen_argmin = min(corner_qs_by_r['unseen(0,+1,-1,0)'], key=corner_qs_by_r['unseen(0,+1,-1,0)'].get)
    print(f'\n6) generalization to unseen r=(0,+1,-1,0): '
          f'peaks at {unseen_argmax} (expect Green), troughs at {unseen_argmin} (expect Blue)')
    generalization_ok = (unseen_argmax == 'Green') and (unseen_argmin == 'Blue')

    action_ok = all(s >= args.action_eps for s in action_std_by_r.values())
    corner_ok = all(s >= args.corner_eps for s in corner_std_by_r.values())

    verdict = 'HEALTHY' if (action_ok and corner_ok and response_ok and range_ok and generalization_ok) else 'FAIL'
    print(f'\n=== VERDICT: {verdict} ===')
    print(f'  action-sensitive (all r): {action_ok}')
    print(f'  state-sensitive (all r):  {corner_ok}')
    print(f'  reward-condition response (A1->Red, A2->Green): {response_ok}')
    print(f'  value-range match (<= {args.range_ratio_eps}x): {range_ok}')
    print(f'  generalization to unseen r: {generalization_ok}')

    return 0 if verdict == 'HEALTHY' else 1


if __name__ == '__main__':
    sys.exit(main())
