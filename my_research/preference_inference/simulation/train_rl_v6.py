"""
v6 S1: reward-conditioned SAC training. Q(s,a,r) / pi(a|s,r), r resampled
per episode. See docs/v6_instructions.md §2.3, §3.0 (S1) and
docs/v6_experiment_log.md §3-4 for the design and its rationale.

Copy-then-extend of train_rl_v3.py (same reasoning as
model/rl_agent_sac_v6.py: keeps my_research/rl_agent_sac.py, shared with
the protected my_research/step3/ line, untouched).

Differences from train_rl_v3.py:
  - actor/critic take a 4-dim reward-parameter r (model/rl_agent_sac_v6.py)
  - r is sampled once per episode (continuous-uniform + L1-normalized,
    sample_reward_param()) and held fixed for that episode's transitions
    (this is the "behavior r" -- the one the actor actually saw and acted
    under)
  - reward is the general 4-landmark form (reward_from_vision), not the
    hardcoded red_fraction - green_fraction
  - ReplayBuffer transitions carry r alongside (s,a,reward,s',done)
  - hindsight relabeling (2026-09-18, user suggestion after S1 trial 1):
    since reward is a pure function of the observed vision (not of which r
    generated the action), each real transition is relabeled with
    K_RELABEL-1 additional resampled r's, recomputing reward for each from
    the SAME (state, action, next_state) -- multiplying (s,a,r) coverage
    per environment step without extra rollout cost. This is the standard
    off-policy relabeling argument (as in Hindsight Experience Replay): a
    transition (s,a,s') is valid under any r, only the reward/done need
    recomputing, and Q-learning is off-policy so training on
    counterfactual r's the actor didn't actually act under is sound.

Run inside Docker:
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python -u my_research/preference_inference/simulation/train_rl_v6.py \
        > /tmp/train_rl_v6.log 2>&1
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
    ActorLSTM, CriticLSTM, R_DIM, sample_reward_param, reward_from_vision,
    A1_TRUE_R, A2_TRUE_R,
)

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPISODES = 1000
MAX_STEPS = 100
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001
LR_ALPHA = 0.0001
BATCH_SIZE = 32
MEMORY_SIZE = 100000
TARGET_ENTROPY = -2.0  # kept at v3/v4's validated value -- v4_experiment_log.md
                        # §7.8 documents that relaxing this caused critic
                        # collapse across all conditions, so this is NOT
                        # touched per the 2026-09-18 decision (see
                        # docs/v6_experiment_log.md)
CURRICULUM_PHASE1_END = 200  # ep<200: r in {A1,A2} only
CURRICULUM_PHASE2_END = 500  # 200<=ep<500: linear mix of {A1,A2} and full-range;
                              # ep>=500: full-range uniform (500 episodes, the
                              # longest phase -- see docs/v6_experiment_log.md §25.1
                              # for why: S5 needs the critic to handle the continuous
                              # r VE' will actually output, not just A1/A2)

K_RELABEL = 10  # hindsight relabeling factor (1 real + 9 relabeled r's per transition):
                # each real env step now pushes 10 transitions, so
                # MEMORY_SIZE=100000 retains ~100 recent episodes' worth
                # (100000 / (100 steps * 10 pushes/step)), vs 500 episodes
                # pre-relabeling (50000/100) -- a deliberate trade-off of
                # per-episode state coverage for far more (s,a,r) coverage
                # per environment step

ENV_CONFIG = '/work/simulation/config/collect/self_random_other_stay.yml'

SAVE_DIR = '/work/my_research/preference_inference/data/model'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vision_to_tensor(vision_np):
    v = vision_np.astype(np.float32)
    v = np.transpose(v, (2, 0, 1))
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def reward_param_to_tensor(w):
    return torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def sample_r_curriculum(episode):
    """docs/v6_experiment_log.md §25.1 (2026-09-28): widen-the-range curriculum,
    NOT a fixed-order (A1-then-A2-then-random) schedule -- the user rejected that
    for two reasons: catastrophic forgetting biases toward whichever phase ran
    last if the final random phase is short, and choosing which agent goes first
    is itself an uncontrolled confound. This schedule instead keeps both A1 and
    A2 in the mix at every point in training (each drawn wp 1/2 whenever the
    "two-point" branch is taken) and is continuous at both phase boundaries
    (100% two-point on both sides of ep=200; ~100% full-range on both sides of
    ep=500), so there is no discontinuity in the sampling distribution to
    confound with the actual widening of difficulty."""
    two_point = [A1_TRUE_R, A2_TRUE_R]
    if episode < CURRICULUM_PHASE1_END:
        return two_point[np.random.randint(2)].copy()
    elif episode < CURRICULUM_PHASE2_END:
        t = (episode - CURRICULUM_PHASE1_END) / float(CURRICULUM_PHASE2_END - CURRICULUM_PHASE1_END)
        if np.random.random() < t:
            return sample_reward_param()
        else:
            return two_point[np.random.randint(2)].copy()
    else:
        return sample_reward_param()


def get_reward_v3(vision_np):
    """Verbatim copy of train_rl_v3.py:get_reward, used only by --control."""
    r, g, b = vision_np[:, :, 0], vision_np[:, :, 1], vision_np[:, :, 2]
    red_pixels = int(((r > 0.9) & (g < 0.1) & (b < 0.1)).sum())
    green_pixels = int(((g > 0.9) & (r < 0.1) & (b < 0.1)).sum())
    total_pixels = vision_np.shape[0] * vision_np.shape[1]
    red_fraction = red_pixels / total_pixels
    green_fraction = green_pixels / total_pixels
    return red_fraction - green_fraction


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


def train(seed, tag='', control=False, no_relabel=False, film=False, curriculum=False):
    """control=True (2026-09-20, docs/v6_experiment_log.md §16): the same
    v6 code path with every r-related element removed, to test whether the
    v6 implementation itself regressed vs v3. r_dim=0 (critic/actor take no
    r, architecture identical to v3's), no r sampling, no relabeling
    (K_RELABEL=1, MEMORY_SIZE back to v3's 50000 since the 100000 was only
    there to compensate relabeling), reward = red_fraction - green_fraction.
    Everything else (env, episodes, steps, GAMMA/TAU/LRs/BATCH_SIZE,
    TARGET_ENTROPY, update loop) is untouched."""
    assert not (control and film), '--control and --film are mutually exclusive'
    assert not (control and curriculum), '--control and --curriculum are mutually exclusive'
    if not tag:
        tag = 'control' if control else (
            'film_norelabel_curriculum' if (film and no_relabel and curriculum) else
            'film' if film else ('norelabel' if no_relabel else ''))
    r_dim = 0 if control else R_DIM
    # --no_relabel (2026-09-21, §17.2): r-conditioned, K_RELABEL=1, MEMORY_SIZE
    # back to 50000 -- the exact trial-1 recipe, to isolate r from relabeling
    memory_size = 50000 if (control or no_relabel) else MEMORY_SIZE
    k_relabel = 1 if (control or no_relabel) else K_RELABEL
    os.makedirs(SAVE_DIR, exist_ok=True)
    seed_all(seed)
    suffix = f'_{tag}' if tag else ''
    actor_save = os.path.join(SAVE_DIR, f'v6_rl_actor_seed{seed}{suffix}.pth')
    critic_save = os.path.join(SAVE_DIR, f'v6_rl_critic_seed{seed}{suffix}.pth')

    config = load_config(ENV_CONFIG)
    env = creator.create_environment(config.environment)
    env.init()
    env.off_display()

    actor = ActorLSTM(r_dim=r_dim, film=film).to(DEVICE)
    critic1 = CriticLSTM(r_dim=r_dim, film=film).to(DEVICE)
    critic2 = CriticLSTM(r_dim=r_dim, film=film).to(DEVICE)
    critic1_target = CriticLSTM(r_dim=r_dim, film=film).to(DEVICE)
    critic2_target = CriticLSTM(r_dim=r_dim, film=film).to(DEVICE)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha = log_alpha.exp()

    actor_opt = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic1_opt = optim.Adam(critic1.parameters(), lr=LR_CRITIC)
    critic2_opt = optim.Adam(critic2.parameters(), lr=LR_CRITIC)
    alpha_opt = optim.Adam([log_alpha], lr=LR_ALPHA)

    memory = ReplayBuffer(memory_size)
    episode_rewards = []

    print(f'Device: {DEVICE}')
    print(f'Seed: {seed}')
    if control:
        print('CONTROL MODE: no r (r_dim=0), no relabeling, MEMORY_SIZE=50000, '
              'reward = red_fraction - green_fraction (v3 reward)')
    else:
        print(f'Variant: film={film} no_relabel={no_relabel} curriculum={curriculum} '
              f'K_RELABEL={k_relabel} MEMORY_SIZE={memory_size}')
        print('Algorithm: SAC (CNN+LSTM, continuous action, reward-conditioned Q(s,a,r)/pi(a|s,r))')
        print('Reward: sum_k w_k * fraction_k, r resampled per episode (continuous-uniform, L1-normalized)')
    print(f'Episodes: {EPISODES} x {MAX_STEPS} steps')

    for episode in range(EPISODES):
        if control:
            r_param_np = np.zeros(0, dtype=np.float32)
        elif curriculum:
            r_param_np = sample_r_curriculum(episode)
        else:
            r_param_np = sample_reward_param()
        r_param_t = reward_param_to_tensor(r_param_np)

        env.reset()
        v_raw, _, _, _, _ = env.step()
        state = vision_to_tensor(v_raw)
        hidden = None
        total_reward = 0.0

        for step in range(MAX_STEPS):
            with torch.no_grad():
                action_t, _, hidden = actor.sample(state, r_param_t, hidden)
            action_np = action_t.squeeze(0).cpu().numpy()

            env.self_agent.p = np.clip(
                env.self_agent.p + action_np, -9.5, 9.5
            )
            v_next_raw, _, _, _, _ = env.step()
            next_state = vision_to_tensor(v_next_raw)

            reward = (get_reward_v3(v_next_raw) if control
                      else reward_from_vision(v_next_raw, r_param_np))
            total_reward += reward
            done = (step == MAX_STEPS - 1)

            memory.push(state, action_t, r_param_np, reward, next_state, done)

            # hindsight relabeling: same (s,a,s') is valid under any r --
            # recompute reward for K_RELABEL-1 other r's and push those too
            for _ in range(k_relabel - 1):
                r_relabel = sample_reward_param()
                reward_relabel = reward_from_vision(v_next_raw, r_relabel)
                memory.push(state, action_t, r_relabel, reward_relabel, next_state, done)

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
                    target_q = rewards_b + GAMMA * (1 - dones) * (
                        torch.min(q1n, q2n) - alpha * nlp)

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
                alpha = log_alpha.exp()

                soft_update(critic1_target, critic1, TAU)
                soft_update(critic2_target, critic2, TAU)

        episode_rewards.append(total_reward)

        if episode % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f'  ep {episode:4d} | avg_reward(50ep): {avg:.3f} | alpha: {alpha.item():.4f} | last_r={r_param_np}')

    torch.save(actor.state_dict(), actor_save)
    torch.save(critic1.state_dict(), critic_save)
    print(f'Saved: {actor_save}')
    print(f'Saved: {critic_save}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default='',
                         help='appended to checkpoint filenames, e.g. "relabel", '
                              'to keep runs from overwriting each other')
    parser.add_argument('--control', action='store_true',
                         help='r-free control run (see train() docstring); '
                              'defaults --tag to "control"')
    parser.add_argument('--no_relabel', action='store_true',
                         help='r-conditioned without relabeling, MEMORY_SIZE=50000 '
                              '(trial-1 recipe, §17.2); defaults --tag to "norelabel"')
    parser.add_argument('--film', action='store_true',
                         help='FiLM-modulate the CNN features by r (§17.3); '
                              'defaults --tag to "film"')
    parser.add_argument('--curriculum', action='store_true',
                         help='widen-the-range r-sampling curriculum (§25.1): '
                              'ep<200 two-point {A1,A2}, 200<=ep<500 linear mix, '
                              'ep>=500 full-range uniform')
    args = parser.parse_args()
    train(args.seed, tag=args.tag, control=args.control,
          no_relabel=args.no_relabel, film=args.film, curriculum=args.curriculum)
