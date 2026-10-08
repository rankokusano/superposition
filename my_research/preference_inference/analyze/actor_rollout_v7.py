"""
v7 stage-0 gate (docs/v7_experiment_log.md §2.1, §2.5): does the v6 seed1
actor reach each of the 4 goals used for A-2 in v7?

Copy of analyze/actor_rollout_v6.py (that script writes into
data/result/v6_baseline/, which is protected, so it is not run). The
rollout loop is unchanged -- same env, same env.reset(), same seeds, same
50ep x 100step -- so the first two conditions (Red, Green) reuse exactly
the method of the v6 run. (v6 §26.1 cannot be reproduced bit for bit:
initial positions use Python's `random`, which v6 did not seed -- seeded
here.) Blue and Cyan are appended after Red and Green.

Changes from v6:
  - 4 goal conditions r = Red (+1,-1,0,0), Green (-1,+1,0,0),
    Blue (0,0,+1,-1), Cyan (0,0,-1,+1); distances to all 4 landmarks.
  - landmark sanity check before the rollouts: render self vision at each
    landmark position and confirm that landmark's colour has the largest
    pixel fraction (checks the coordinates, which live in field.obj).
  - the actor-health block (A1 vs A2 action means) is dropped; it is not
    part of the gate and ran after the rollouts, so RNG order is unaffected.

Usage (inside Docker):
    cd /work/my_research/preference_inference
    xvfb-run --auto-servernum python3 analyze/actor_rollout_v7.py \
        --actor_path data/model/v6_rl_actor_seed1_film_norelabel_curriculum.pth \
        --label seed1_film_norelabel_curriculum --film
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')

import numpy as np
import torch

import creator
from util import load_config
from my_research.preference_inference.model.rl_agent_sac_v6 import (
    ActorLSTM, landmark_fractions,
)

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENV_CONFIG = '/work/simulation/config/collect/self_random_other_stay.yml'
SAVE_DIR = '/work/my_research/preference_inference/data/result/v7_irl/stage0'
N_EPISODES = 50
MAX_STEPS = 100
SEED = 0
GATE_DIST = 5.0

LANDMARKS = {
    'Red': np.array([-9.0, 9.0]),
    'Green': np.array([-9.0, -9.0]),
    'Blue': np.array([9.0, -9.0]),
    'Cyan': np.array([9.0, 9.0]),
}
LANDMARK_INDEX = {'Red': 0, 'Green': 1, 'Blue': 2, 'Cyan': 3}  # landmark_fractions order
GOALS = {
    'Red': np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32),
    'Green': np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    'Blue': np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32),
    'Cyan': np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32),
}


def vision_to_tensor(v):
    v = v.astype(np.float32)
    return torch.tensor(np.transpose(v, (2, 0, 1))).unsqueeze(0).to(DEVICE)


def r_to_tensor(w):
    return torch.tensor(np.asarray(w, dtype=np.float32)).unsqueeze(0).to(DEVICE)


def rollout(env, actor, r_t, n_episodes, max_steps):
    final_d = {n: [] for n in LANDMARKS}
    min_d = {n: [] for n in LANDMARKS}
    init_d = {n: [] for n in LANDMARKS}
    for ep in range(n_episodes):
        env.reset()
        v_raw, _, _, _, _ = env.step()
        state = vision_to_tensor(v_raw)
        hidden = None
        dists = {n: [np.linalg.norm(env.self_agent.p - lp)] for n, lp in LANDMARKS.items()}
        for n in LANDMARKS:
            init_d[n].append(dists[n][0])
        for step in range(max_steps):
            with torch.no_grad():
                action_t, _, hidden = actor.sample(state, r_t, hidden)
            action_np = action_t.squeeze(0).cpu().numpy()
            env.self_agent.p = np.clip(env.self_agent.p + action_np, -9.5, 9.5)
            v_next_raw, _, _, _, _ = env.step()
            state = vision_to_tensor(v_next_raw)
            for n, lp in LANDMARKS.items():
                dists[n].append(np.linalg.norm(env.self_agent.p - lp))
        for n in LANDMARKS:
            final_d[n].append(dists[n][-1])
            min_d[n].append(min(dists[n]))
    return final_d, min_d, init_d


def landmark_sanity(env):
    """Render self vision with A-1 standing 1.5 units inside each corner
    (A-2 parked at the centre) and return the 4 colour fractions."""
    env.world.set_camera('self')
    env.other_agent.p = np.array([0.0, 0.0])
    out = {}
    for name, lp in LANDMARKS.items():
        env.self_agent.p = lp * (7.5 / 9.0)
        env.world.draw(env.self_agent, env.other_agent)
        frac = landmark_fractions(env.world.capture())
        out[name] = dict(fractions=[float(x) for x in frac],
                         argmax=list(LANDMARK_INDEX)[int(np.argmax(frac))])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_path', required=True)
    parser.add_argument('--label', default='actor')
    parser.add_argument('--film', action='store_true')
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()

    env_cfg = load_config(ENV_CONFIG)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()

    actor = ActorLSTM(film=args.film).to(DEVICE)
    actor.load_state_dict(torch.load(args.actor_path, map_location=DEVICE), strict=True)
    actor.eval()

    # actor_rollout_v6.py seeded torch and numpy only, but initial positions
    # come from Python's `random` (simulation/simulation/agent/field_object.py),
    # so v6 rollouts were not reproducible. Seed it too. The sanity render
    # below draws no random numbers (fixed positions).
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    sanity = landmark_sanity(env)
    print('=== landmark sanity (fractions red, green, blue, cyan) ===')
    for name, s in sanity.items():
        print(f'  near {name:5s}: {np.round(s["fractions"], 4)}  argmax={s["argmax"]}')

    results = {'actor_path': args.actor_path, 'n_episodes': N_EPISODES,
               'max_steps': MAX_STEPS, 'seed': SEED, 'gate_dist': GATE_DIST,
               'landmark_sanity': sanity, 'goals': {}}
    print(f'=== actor rollout: {args.label} ({args.actor_path}) ===')
    for goal, r_vec in GOALS.items():
        final_d, min_d, init_d = rollout(env, actor, r_to_tensor(r_vec), N_EPISODES, MAX_STEPS)
        g = dict(
            r=[float(x) for x in r_vec],
            init_target=float(np.mean(init_d[goal])),
            final_target=float(np.mean(final_d[goal])), final_target_sd=float(np.std(final_d[goal])),
            min_target=float(np.mean(min_d[goal])), min_target_sd=float(np.std(min_d[goal])),
            final_all={n: float(np.mean(final_d[n])) for n in LANDMARKS},
            frac_episodes_final_within_gate=float(np.mean(np.array(final_d[goal]) <= GATE_DIST)),
        )
        g['pass'] = bool(g['final_target'] <= GATE_DIST)
        results['goals'][goal] = g
        print(f'  goal={goal:5s}: init={g["init_target"]:.2f}  final={g["final_target"]:.2f}'
              f'+-{g["final_target_sd"]:.2f}  min={g["min_target"]:.2f}+-{g["min_target_sd"]:.2f}  '
              f'final_all={ {k: round(v, 2) for k, v in g["final_all"].items()} }  '
              f'pass={g["pass"]}')

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'actor_rollout_{args.label}{args.suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
