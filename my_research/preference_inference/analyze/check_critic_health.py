"""
Critic health check -- mandatory after every SAC (re)training run from here
on (per 2026-08-12 decision).

Detects a "collapsed" critic: one whose output has become (near-)constant
regardless of state (vision) and/or action, which happened to A-1's critic
after the continuous-reward retrain (Sec.5.2 revision) despite the actor
looking fine. A collapsed critic makes any downstream use of Q(s,a) --
which is this whole project's central design element -- meaningless, so
this needs to be checked before trusting any other result from a run.

Checks, in order of how damning a failure is:
  1. action sensitivity: same real vision, several different probe actions
     -> Q should vary meaningfully across them.
  2. state sensitivity (real): same action, vision from the 4 landmark
     corners + center -> Q should differ across genuinely different scenes.
  3. state sensitivity (out-of-distribution): same action, synthetic vision
     (all-zero / all-one / random noise) -> Q for these SHOULD differ from
     real vision (and from each other) if the critic is actually using its
     vision input; a healthy critic is not guaranteed to behave sanely on
     OOD input, but a value that's suspiciously identical to the
     in-distribution value is the strongest collapse signal, since it means
     the vision branch isn't contributing at all.

Reports a PASS/FAIL verdict from simple spread thresholds (tune via
--action-eps / --corner-eps if these prove too strict/loose in practice).

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/check_critic_health.py \
        --critic_path data/model/v3_rl_critic_seed0.pth \
        --env_config /work/simulation/config/collect/self_random_other_stay.yml \
        --camera self --scan_self --label a1_seed0
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
from my_research.rl_agent_sac import CriticLSTM  # noqa

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CORNERS = {
    'Red':    (-9.0,  9.0),
    'Green':  (-9.0, -9.0),
    'Blue':   ( 9.0, -9.0),
    'Yellow': ( 9.0,  9.0),
    'Center': ( 0.0,  0.0),
}

PROBE_ACTIONS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.7, -0.7)]


def vision_to_tensor(v):
    v = v.astype(np.float32)
    return torch.tensor(np.transpose(v, (2, 0, 1))).unsqueeze(0).to(DEVICE)


def q_of(critic, v_t, action):
    a_t = torch.tensor([action], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        q, _ = critic(v_t, a_t, hidden=None)
    return q.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic_path', required=True)
    parser.add_argument('--env_config', required=True)
    parser.add_argument('--camera', choices=['self', 'other'], required=True)
    parser.add_argument('--scan_self', action='store_true',
                         help='move self_agent for the scan (else moves other_agent)')
    parser.add_argument('--label', default='critic')
    parser.add_argument('--action-eps', type=float, default=0.05,
                         help='min std of Q across probe actions to call it action-sensitive')
    parser.add_argument('--corner-eps', type=float, default=0.05,
                         help='min std of Q across the 4 corners+center to call it state-sensitive')
    args = parser.parse_args()

    env_cfg = load_config(args.env_config)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()
    env.reset()
    # fix the non-scanned agent so scenes only vary in the scanned agent's position
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

    print(f'=== critic health check: {args.label} ({args.critic_path}) ===')

    # 1. action sensitivity
    v = render_at(CORNERS['Red'])
    v_t = vision_to_tensor(v)
    action_qs = [q_of(critic, v_t, a) for a in PROBE_ACTIONS]
    action_std = float(np.std(action_qs))
    print(f'\n1) action sensitivity (fixed vision @ Red, {len(PROBE_ACTIONS)} probe actions):')
    for a, q in zip(PROBE_ACTIONS, action_qs):
        print(f'   action={a}  Q={q:.6f}')
    print(f'   std across actions: {action_std:.6f}  (threshold: {args.action_eps})')

    # 2. state sensitivity (real corners)
    print(f'\n2) state sensitivity (fixed action=(1,0), 4 corners + center):')
    corner_qs = {}
    for name, p in CORNERS.items():
        v = render_at(p)
        v_t = vision_to_tensor(v)
        q = q_of(critic, v_t, (1.0, 0.0))
        corner_qs[name] = q
        print(f'   {name:7s} p={p}  Q={q:.6f}')
    corner_std = float(np.std(list(corner_qs.values())))
    print(f'   std across corners: {corner_std:.6f}  (threshold: {args.corner_eps})')

    # 3. OOD sensitivity
    print(f'\n3) out-of-distribution sensitivity (fixed action=(1,0)):')
    real_v = vision_to_tensor(render_at(CORNERS['Red']))
    ood_inputs = {
        'real(Red)': real_v,
        'zeros': torch.zeros_like(real_v),
        'ones': torch.ones_like(real_v),
        'random': torch.rand_like(real_v),
    }
    ood_qs = {}
    for name, v_t in ood_inputs.items():
        q = q_of(critic, v_t, (1.0, 0.0))
        ood_qs[name] = q
        print(f'   vision={name:10s}  Q={q:.6f}')
    ood_std = float(np.std([q for n, q in ood_qs.items() if n != 'real(Red)']))
    print(f'   std across synthetic(zeros/ones/random): {ood_std:.6f}')
    print(f'   |Q(real) - Q(zeros)|: {abs(ood_qs["real(Red)"] - ood_qs["zeros"]):.6f}')

    # verdict
    action_ok = action_std >= args.action_eps
    corner_ok = corner_std >= args.corner_eps
    verdict = 'HEALTHY' if (action_ok and corner_ok) else 'COLLAPSED'
    print(f'\n=== VERDICT: {verdict} ===')
    print(f'  action-sensitive: {action_ok} (std={action_std:.6f})')
    print(f'  state-sensitive:  {corner_ok} (std={corner_std:.6f})')
    if verdict == 'COLLAPSED':
        print('  -> Q(s,a) does not meaningfully depend on state and/or action. '
              'Do not use this critic for anything downstream.')

    return 0 if verdict == 'HEALTHY' else 1


if __name__ == '__main__':
    sys.exit(main())
