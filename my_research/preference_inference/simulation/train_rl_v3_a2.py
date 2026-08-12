"""
v3 RL agent training for A-2: learns green-preference via SAC.

Design:
    - A-2 (other_agent) is controlled by SAC policy
    - A-1 (self_agent) stays stationary during A-2 training
    - Reward (v4 Sec.5.2, revised 2026-08-12):
          reward = green_fraction - red_fraction   (continuous, no threshold)
      Sign-flipped mirror of A-1's red_fraction - green_fraction in
      train_rl_v3.py, symmetric in the same *principle* (vision pixel
      signal, no privileged position). The earlier threshold version
      (green>50px:+1, red>50px:-1) was measured (probe scan at fixed x=-9,
      varying y) to be locally flat across the whole region satisfying
      both conditions -- Q(s,a_probe) would carry no spatial information
      inside that region, which R4/R5 need. Continuous fractions restore
      a gradient. Coverage (the original problem this reward redesign
      was trying to fix) is instead restored by keeping action generation
      stochastic at rollout/collection time -- deterministic tanh(mean)
      rollout measurably collapsed A-2's position std vs. the original
      stochastic-sampling collection, so it is not used here or downstream.
    - Uses 'other' camera to capture A-2's own vision

Run inside Docker:
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python -u my_research/preference_inference/simulation/train_rl_v3_a2.py --seed 0 \
        > /tmp/train_rl_v3_a2.log 2>&1
"""

import argparse
import os
import sys

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

import creator
from util import load_config
from my_research.rl_agent_sac import ActorLSTM, CriticLSTM

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPISODES        = 1000
MAX_STEPS       = 100
GAMMA           = 0.99
TAU             = 0.005
LR_ACTOR        = 0.0001
LR_CRITIC       = 0.0001
LR_ALPHA        = 0.0001
BATCH_SIZE      = 32
MEMORY_SIZE     = 50000
TARGET_ENTROPY  = -2.0  # SAC entropy coefficient target; tune here if coverage needs adjusting

ENV_CONFIG  = '/work/simulation/config/collect/self_stay_other_random.yml'

SAVE_DIR    = '/work/my_research/preference_inference/data/model'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_reward(vision_np):
    r, g, b = vision_np[:, :, 0], vision_np[:, :, 1], vision_np[:, :, 2]
    green_pixels = int(((g > 0.9) & (r < 0.1) & (b < 0.1)).sum())
    red_pixels   = int(((r > 0.9) & (g < 0.1) & (b < 0.1)).sum())
    total_pixels = vision_np.shape[0] * vision_np.shape[1]
    green_fraction = green_pixels / total_pixels
    red_fraction = red_pixels / total_pixels
    return green_fraction - red_fraction


def vision_to_tensor(vision_np):
    v = vision_np.astype(np.float32)
    v = np.transpose(v, (2, 0, 1))
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def capture_a2_vision(env):
    """Capture vision from A-2 (other_agent) perspective."""
    env.world.set_camera('other')
    env.world.draw(env.self_agent, env.other_agent)
    return env.world.capture()


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, n):
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)


def train(seed):
    os.makedirs(SAVE_DIR, exist_ok=True)
    seed_all(seed)
    actor_save = os.path.join(SAVE_DIR, f'v3_rl_a2_actor_seed{seed}.pth')
    critic_save = os.path.join(SAVE_DIR, f'v3_rl_a2_critic_seed{seed}.pth')

    config = load_config(ENV_CONFIG)
    env = creator.create_environment(config.environment)
    env.init()
    env.off_display()

    actor          = ActorLSTM().to(DEVICE)
    critic1        = CriticLSTM().to(DEVICE)
    critic2        = CriticLSTM().to(DEVICE)
    critic1_target = CriticLSTM().to(DEVICE)
    critic2_target = CriticLSTM().to(DEVICE)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha = log_alpha.exp()

    actor_opt   = optim.Adam(actor.parameters(),   lr=LR_ACTOR)
    critic1_opt = optim.Adam(critic1.parameters(), lr=LR_CRITIC)
    critic2_opt = optim.Adam(critic2.parameters(), lr=LR_CRITIC)
    alpha_opt   = optim.Adam([log_alpha],           lr=LR_ALPHA)

    memory = ReplayBuffer(MEMORY_SIZE)
    episode_rewards = []

    print(f'Device: {DEVICE}')
    print(f'Seed: {seed}')
    print(f'Algorithm: SAC (CNN+LSTM, continuous action)')
    print(f'Reward: green_fraction - red_fraction (continuous)')
    print(f'Episodes: {EPISODES} × {MAX_STEPS} steps')

    for episode in range(EPISODES):
        env.reset()
        v_raw = capture_a2_vision(env)
        state = vision_to_tensor(v_raw)
        hidden = None
        total_reward = 0.0

        for step in range(MAX_STEPS):
            with torch.no_grad():
                action_t, _, hidden = actor.sample(state, hidden)
            action_np = action_t.squeeze(0).cpu().numpy()

            # Move A-2 with RL action; A-1 stays stationary
            env.other_agent.p = np.clip(
                env.other_agent.p + action_np, -9.5, 9.5
            )

            v_next_raw = capture_a2_vision(env)
            next_state = vision_to_tensor(v_next_raw)

            reward = get_reward(v_next_raw)
            total_reward += reward
            done = (step == MAX_STEPS - 1)

            memory.push(state, action_t, reward, next_state, done)
            state = next_state

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards_b, next_states, dones = zip(*batch)

                states      = torch.cat(states)
                actions     = torch.cat(actions)
                rewards_b   = torch.FloatTensor(rewards_b).unsqueeze(1).to(DEVICE)
                next_states = torch.cat(next_states)
                dones       = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

                with torch.no_grad():
                    na, nlp, _ = actor.sample(next_states)
                    q1n, _ = critic1_target(next_states, na)
                    q2n, _ = critic2_target(next_states, na)
                    target_q = rewards_b + GAMMA * (1 - dones) * (
                        torch.min(q1n, q2n) - alpha * nlp)

                q1, _ = critic1(states, actions)
                q2, _ = critic2(states, actions)
                critic1_opt.zero_grad()
                nn.MSELoss()(q1, target_q).backward()
                critic1_opt.step()

                critic2_opt.zero_grad()
                nn.MSELoss()(q2, target_q).backward()
                critic2_opt.step()

                na2, lp2, _ = actor.sample(states)
                q1a, _ = critic1(states, na2)
                q2a, _ = critic2(states, na2)
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
            print(f'  ep {episode:4d} | avg_reward(50ep): {avg:.3f} | alpha: {alpha.item():.4f}')

    torch.save(actor.state_dict(), actor_save)
    torch.save(critic1.state_dict(), critic_save)
    print(f'Saved: {actor_save}')
    print(f'Saved: {critic_save}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    train(args.seed)
