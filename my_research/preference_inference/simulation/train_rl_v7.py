"""
v7 retraining of the reward-conditioned SAC actor/critic
(docs/v7_experiment_log.md §2.11). New script; simulation/train_rl_v6.py and
the v6 checkpoints are left untouched.

The SAC update is the v6 one (train_rl_v6.py, FiLM, no relabeling,
MEMORY_SIZE 50000, same GAMMA/TAU/LRs/BATCH_SIZE/TARGET_ENTROPY). Two
things change:

1. Environment = the agent's own first-person experience, laid out as in
   collect_data_v7.py: the learning agent moves ONLY by its own action
   (clipped to the arena boundary, +-10), while the other agent random-walks
   in view. In train_rl_v6.py the learner was env.self_agent and env.step()
   added a RandomAgent step (v=1) to it after every action (§8); here the
   learner is env.other_agent (camera 'other') and env.self_agent is the
   random walker, so no random step is added to the learner. Per step: the
   learner acts from its current view, moves, the other agent takes one
   random-walk step, then the learner's next view is captured (same order
   as collect_data_v7.py).

2. r sampling (--sampling):
   named : 8 named goals (Red, Green, Cyan, Blue, Mtop, Mleft, Mbottom,
           Mright; uniform) mixed with full-range sample_reward_param().
           ep<300 named only; 300<=ep<700 named share falls linearly 1 -> 0.5;
           ep>=700 named share 0.5.                      (condition A)
   named_mixheavy (remake 1, §2.13): same 8 named goals, mixed goals 2/3
           of the named mass; named only for ep<500, share 1 -> 0.7 over
           500-1000, then 0.7.  --alpha_min clamps alpha from below.
   v6    : v6's curriculum unchanged (ep<200 {A1,A2}, 200-500 linear mix,
           ep>=500 full-range).                          (condition B)

The critic used for inference (A-1) and the actor that generates the data
(A-2) are the same recipe with different seeds; the critic only ever learns
from its own agent's experience.

Run inside Docker (from /work/my_research/preference_inference):
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python -u simulation/train_rl_v7.py --sampling named --episodes 2000 --seed 0 --tag condA
"""

import argparse
import os
import random
import sys
from collections import deque

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import creator
from util import load_config
from my_research.preference_inference.model.rl_agent_sac_v6 import (
    ActorLSTM, CriticLSTM, sample_reward_param, reward_from_vision, A1_TRUE_R, A2_TRUE_R,
)

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_STEPS = 100
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001
LR_ALPHA = 0.0001
BATCH_SIZE = 32
MEMORY_SIZE = 50000          # v6 no-relabel recipe
TARGET_ENTROPY = -2.0        # unchanged from v3/v4/v6
SAVE_EVERY = 500             # record-only intermediate checkpoints; the final one is used

ENV_CONFIG = '/work/simulation/config/collect/self_random_other_stay.yml'
SAVE_DIR = '/work/my_research/preference_inference/data/model/v7'

NAMED_GOALS = {
    'Red': [1, -1, 0, 0], 'Green': [-1, 1, 0, 0], 'Cyan': [0, 0, -1, 1], 'Blue': [0, 0, 1, -1],
    'Mtop': [1, 0, 0, 1], 'Mleft': [1, 1, 0, 0], 'Mbottom': [0, 1, 1, 0], 'Mright': [0, 0, 1, 1],
}
NAMED_R = np.array(list(NAMED_GOALS.values()), dtype=np.float32)
NAMED_ONLY_END = 300
NAMED_RAMP_END = 700
NAMED_SHARE_FINAL = 0.5
# remake 1 (docs/v7_experiment_log.md §2.13): mixed goals weighted 2/3 within the named goals
MIXHEAVY_P = np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.float64) / 12.0  # order of NAMED_GOALS
MIXHEAVY_ONLY_END, MIXHEAVY_RAMP_END, MIXHEAVY_SHARE_FINAL = 500, 1000, 0.7
V6_PHASE1_END, V6_PHASE2_END = 200, 500


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vision_to_tensor(vision_np):
    v = np.transpose(vision_np.astype(np.float32), (2, 0, 1))
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def named_share(episode):
    if episode < NAMED_ONLY_END:
        return 1.0
    if episode < NAMED_RAMP_END:
        t = (episode - NAMED_ONLY_END) / float(NAMED_RAMP_END - NAMED_ONLY_END)
        return 1.0 - t * (1.0 - NAMED_SHARE_FINAL)
    return NAMED_SHARE_FINAL


def mixheavy_share(episode):
    if episode < MIXHEAVY_ONLY_END:
        return 1.0
    if episode < MIXHEAVY_RAMP_END:
        t = (episode - MIXHEAVY_ONLY_END) / float(MIXHEAVY_RAMP_END - MIXHEAVY_ONLY_END)
        return 1.0 - t * (1.0 - MIXHEAVY_SHARE_FINAL)
    return MIXHEAVY_SHARE_FINAL


def sample_r(episode, sampling):
    """Returns (r, label) where label names the named goal or 'range'."""
    if sampling == 'named_mixheavy':
        if np.random.random() < mixheavy_share(episode):
            k = np.random.choice(len(NAMED_R), p=MIXHEAVY_P)
            return NAMED_R[k].copy(), list(NAMED_GOALS)[k]
        return sample_reward_param(), 'range'
    if sampling == 'named':
        if np.random.random() < named_share(episode):
            k = np.random.randint(len(NAMED_R))
            return NAMED_R[k].copy(), list(NAMED_GOALS)[k]
        return sample_reward_param(), 'range'
    # v6 curriculum, verbatim logic of train_rl_v6.sample_r_curriculum
    two_point = [A1_TRUE_R, A2_TRUE_R]
    if episode < V6_PHASE1_END:
        return two_point[np.random.randint(2)].copy(), 'two_point'
    if episode < V6_PHASE2_END:
        t = (episode - V6_PHASE1_END) / float(V6_PHASE2_END - V6_PHASE1_END)
        if np.random.random() < t:
            return sample_reward_param(), 'range'
        return two_point[np.random.randint(2)].copy(), 'two_point'
    return sample_reward_param(), 'range'


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r_param, reward, ns, done):
        self.buffer.append((s, a, r_param, reward, ns, done))

    def sample(self, n):
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)


def capture_learner(env):
    env.world.set_camera('other')
    env.world.draw(env.self_agent, env.other_agent)
    return env.world.capture()


def train(seed, sampling, episodes, tag, alpha_min=None):
    os.makedirs(SAVE_DIR, exist_ok=True)
    seed_all(seed)
    name = f'seed{seed}_{tag}'
    actor_save = os.path.join(SAVE_DIR, f'v7_rl_actor_{name}.pth')
    critic_save = os.path.join(SAVE_DIR, f'v7_rl_critic_{name}.pth')
    for p in (actor_save, critic_save):
        if os.path.exists(p):
            raise RuntimeError(f'{p} exists; refusing to overwrite')

    env = creator.create_environment(load_config(ENV_CONFIG).environment)
    env.init()
    env.off_display()
    bound = env.world.get_boundary()
    lo, hi = np.array([bound[0][0], bound[1][0]]), np.array([bound[0][1], bound[1][1]])

    actor = ActorLSTM(film=True).to(DEVICE)
    critic1 = CriticLSTM(film=True).to(DEVICE)
    critic2 = CriticLSTM(film=True).to(DEVICE)
    critic1_target = CriticLSTM(film=True).to(DEVICE)
    critic2_target = CriticLSTM(film=True).to(DEVICE)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha = log_alpha.exp()
    actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic1_opt = optim.Adam(critic1.parameters(), lr=LR_CRITIC)
    critic2_opt = optim.Adam(critic2.parameters(), lr=LR_CRITIC)
    alpha_opt = optim.Adam([log_alpha], lr=LR_ALPHA)
    memory = ReplayBuffer(MEMORY_SIZE)
    episode_rewards, labels = [], []

    print(f'Device: {DEVICE}  Seed: {seed}  sampling={sampling}  episodes={episodes}  tag={tag}')
    print('Env: learner = env.other_agent (own action only, clip +-10, no random step); '
          'env.self_agent random-walks in view (collect_data_v7.py layout)')
    print(f'FiLM=True  relabel=False  MEMORY_SIZE={MEMORY_SIZE}  TARGET_ENTROPY={TARGET_ENTROPY}  alpha_min={alpha_min}')

    for episode in range(episodes):
        r_np, label = sample_r(episode, sampling)
        labels.append(label)
        r_t = torch.tensor(r_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        env.reset()
        env.self_agent.step()
        state = vision_to_tensor(capture_learner(env))
        hidden = None
        total_reward = 0.0

        for step in range(MAX_STEPS):
            with torch.no_grad():
                action_t, _, hidden = actor.sample(state, r_t, hidden)
            action_np = action_t.squeeze(0).cpu().numpy()
            env.other_agent.p = np.clip(env.other_agent.p + action_np, lo, hi)
            env.self_agent.step()
            v_next = capture_learner(env)
            next_state = vision_to_tensor(v_next)
            reward = reward_from_vision(v_next, r_np)
            total_reward += reward
            done = (step == MAX_STEPS - 1)
            memory.push(state, action_t, r_np, reward, next_state, done)
            state = next_state

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, r_params, rewards_b, next_states, dones = zip(*batch)
                states = torch.cat(states)
                actions = torch.cat(actions)
                r_params_t = torch.tensor(np.stack(r_params), dtype=torch.float32).to(DEVICE)
                rewards_b = torch.FloatTensor(rewards_b).unsqueeze(1).to(DEVICE)
                next_states = torch.cat(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

                with torch.no_grad():
                    na, nlp, _ = actor.sample(next_states, r_params_t)
                    q1n, _ = critic1_target(next_states, na, r_params_t)
                    q2n, _ = critic2_target(next_states, na, r_params_t)
                    target_q = rewards_b + GAMMA * (1 - dones) * (torch.min(q1n, q2n) - alpha * nlp)

                q1, _ = critic1(states, actions, r_params_t)
                q2, _ = critic2(states, actions, r_params_t)
                critic1_opt.zero_grad()
                nn.MSELoss()(q1, target_q).backward()
                critic1_opt.step()
                critic2_opt.zero_grad()
                nn.MSELoss()(q2, target_q).backward()
                critic2_opt.step()

                na2, lp2, _ = actor.sample(states, r_params_t)
                q1a, _ = critic1(states, na2, r_params_t)
                q2a, _ = critic2(states, na2, r_params_t)
                actor_opt.zero_grad()
                (alpha * lp2 - torch.min(q1a, q2a)).mean().backward()
                actor_opt.step()

                alpha_opt.zero_grad()
                (-(log_alpha * (lp2 + TARGET_ENTROPY).detach())).mean().backward()
                alpha_opt.step()
                if alpha_min is not None:  # remake 1 (§2.13): alpha >= alpha_min
                    with torch.no_grad():
                        log_alpha.clamp_(min=float(np.log(alpha_min)))
                alpha = log_alpha.exp()

                soft_update(critic1_target, critic1, TAU)
                soft_update(critic2_target, critic2, TAU)

        episode_rewards.append(total_reward)
        if episode % 50 == 0:
            recent = labels[-50:]
            share = sum(l not in ('range',) for l in recent) / len(recent)
            print(f'  ep {episode:4d} | avg_reward(50ep): {np.mean(episode_rewards[-50:]):.3f} | '
                  f'alpha: {alpha.item():.4f} | named/two-point share(50ep): {share:.2f} | '
                  f'last r={np.round(r_np, 3)} ({label})', flush=True)
        if (episode + 1) % SAVE_EVERY == 0 and (episode + 1) < episodes:
            torch.save(actor.state_dict(), actor_save.replace('.pth', f'_ep{episode + 1}.pth'))
            torch.save(critic1.state_dict(), critic_save.replace('.pth', f'_ep{episode + 1}.pth'))

    torch.save(actor.state_dict(), actor_save)
    torch.save(critic1.state_dict(), critic_save)
    counts = {k: labels.count(k) for k in sorted(set(labels))}
    print(f'r label counts over training: {counts}')
    print(f'Saved: {actor_save}')
    print(f'Saved: {critic_save}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--sampling', choices=['named', 'named_mixheavy', 'v6'], required=True)
    ap.add_argument('--episodes', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--alpha_min', type=float, default=None, help='remake 1: lower bound on alpha (0.002)')
    a = ap.parse_args()
    train(a.seed, a.sampling, a.episodes, a.tag, alpha_min=a.alpha_min)
