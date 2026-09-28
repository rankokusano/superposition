"""
Does the trained actor actually respond to r by visiting the preferred
landmark, or does it ignore r (docs/v6_experiment_log.md §24, user
hypothesis after the S1 2x2 deconfound pointed at "r" not "data")?

Method 2 (primary): roll out the trained ActorLSTM for N_EPISODES under
r=A1 (fixed) and r=A2 (fixed), MAX_STEPS each, using env.reset() with no
position override -- exactly matching train_rl_v6.py's training rollout.
Track self_agent.p each step, report mean final/min distance to the
target landmark (Red for A1, Green for A2) and to the "wrong" landmark,
for selectivity.

Method 3: actor action-mean difference between r=A1 and r=A2 at a fixed
set of states (vision frames sampled across the arena).

Usage (inside Docker):
    cd /work/my_research/preference_inference
    xvfb-run --auto-servernum python3 analyze/actor_rollout_v6.py \
        --actor_path data/model/v6_rl_actor_seed0_film_norelabel.pth --label film_norelabel_s0 --film
"""
import argparse
import json
import os
import sys

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')

import numpy as np
import torch

import creator
from util import load_config
from my_research.preference_inference.model.rl_agent_sac_v6 import (
    ActorLSTM, A1_TRUE_R, A2_TRUE_R,
)

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENV_CONFIG = '/work/simulation/config/collect/self_random_other_stay.yml'
SAVE_DIR = '/work/my_research/preference_inference/data/result/v6_baseline'
N_EPISODES = 50
MAX_STEPS = 100
SEED = 0

LANDMARKS = {'Red': np.array([-9.0, 9.0]), 'Green': np.array([-9.0, -9.0])}
R_CONDITIONS = {'A1': (A1_TRUE_R, 'Red', 'Green'), 'A2': (A2_TRUE_R, 'Green', 'Red')}


def vision_to_tensor(v):
    v = v.astype(np.float32)
    return torch.tensor(np.transpose(v, (2, 0, 1))).unsqueeze(0).to(DEVICE)


def r_to_tensor(w):
    return torch.tensor(np.asarray(w, dtype=np.float32)).unsqueeze(0).to(DEVICE)


def rollout(env, actor, r_t, n_episodes, max_steps, rng):
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


@torch.no_grad()
def actor_health(env, actor):
    rng = np.random.RandomState(SEED)
    env.world.set_camera('self')
    env.other_agent.p = np.array([0.0, 0.0])
    diffs = []
    corrs = []
    for _ in range(30):
        p = rng.uniform(-9.5, 9.5, size=2)
        env.self_agent.p = p
        env.world.draw(env.self_agent, env.other_agent)
        v = vision_to_tensor(env.world.capture())
        m1, _, _ = actor.forward(v, r_to_tensor(A1_TRUE_R))
        m2, _, _ = actor.forward(v, r_to_tensor(A2_TRUE_R))
        m1, m2 = m1.cpu().numpy()[0], m2.cpu().numpy()[0]
        diffs.append(float(np.linalg.norm(m1 - m2)))
        corrs.append(float(np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-8)))
    return dict(mean_L2_diff=float(np.mean(diffs)), mean_cos=float(np.mean(corrs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_path', required=True)
    parser.add_argument('--label', default='actor')
    parser.add_argument('--film', action='store_true')
    args = parser.parse_args()

    env_cfg = load_config(ENV_CONFIG)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()

    actor = ActorLSTM(film=args.film).to(DEVICE)
    actor.load_state_dict(torch.load(args.actor_path, map_location=DEVICE), strict=True)
    actor.eval()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    results = {}
    print(f'=== actor rollout: {args.label} ({args.actor_path}) ===')
    for cond, (r_vec, target, other) in R_CONDITIONS.items():
        final_d, min_d, init_d = rollout(env, actor, r_to_tensor(r_vec), N_EPISODES, MAX_STEPS, rng)
        results[cond] = dict(
            target=target, other=other,
            init_target=float(np.mean(init_d[target])),
            final_target=float(np.mean(final_d[target])), final_target_sd=float(np.std(final_d[target])),
            final_other=float(np.mean(final_d[other])),
            min_target=float(np.mean(min_d[target])), min_target_sd=float(np.std(min_d[target])),
            min_other=float(np.mean(min_d[other])),
        )
        print(f'  r={cond} (target={target}): init_dist(target)={results[cond]["init_target"]:.2f}  '
              f'final_dist(target)={results[cond]["final_target"]:.2f}'
              f'+-{results[cond]["final_target_sd"]:.2f}  final_dist(other={other})={results[cond]["final_other"]:.2f}  '
              f'min_dist(target)={results[cond]["min_target"]:.2f}+-{results[cond]["min_target_sd"]:.2f}  '
              f'min_dist(other)={results[cond]["min_other"]:.2f}')

    health = actor_health(env, actor)
    results['actor_health'] = health
    print(f'  actor r-response (30 fixed states): mean|mean(A1)-mean(A2)|={health["mean_L2_diff"]:.4f} '
          f'  mean cos(mean(A1),mean(A2))={health["mean_cos"]:.4f}')

    out_path = os.path.join(SAVE_DIR, f'actor_rollout_{args.label}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
