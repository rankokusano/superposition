"""
v3 RL agent training for A-2: learns green-preference via SAC.

Design:
    - A-2 (other_agent) is controlled by SAC policy
    - A-1 (self_agent) stays stationary during A-2 training
    - Reward: green visible (>50 px) → +1.0 only
    - Uses 'other' camera to capture A-2's own vision

Run inside Docker:
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python -u my_research/preference_inference/simulation/train_rl_v3_a2.py \
        > /tmp/train_rl_v3_a2.log 2>&1
"""

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
TARGET_ENTROPY  = -2.0
GREEN_THRESHOLD = 50

ENV_CONFIG  = '/work/simulation/config/collect/self_stay_other_random.yml'

SAVE_DIR    = '/work/my_research/preference_inference/data/model'
ACTOR_SAVE  = os.path.join(SAVE_DIR, 'v3_rl_a2_actor.pth')
CRITIC_SAVE = os.path.join(SAVE_DIR, 'v3_rl_a2_critic.pth')


def get_reward(vision_np):
    r, g, b = vision_np[:, :, 0], vision_np[:, :, 1], vision_np[:, :, 2]
    green_pixels = int(((g > 0.9) & (r < 0.1) & (b < 0.1)).sum())
    return 1.0 if green_pixels > GREEN_THRESHOLD else 0.0


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


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

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
    print(f'Algorithm: SAC (CNN+LSTM, continuous action)')
    print(f'Reward: green >50px → +1.0 only')
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

    torch.save(actor.state_dict(), ACTOR_SAVE)
    torch.save(critic1.state_dict(), CRITIC_SAVE)
    print(f'Saved: {ACTOR_SAVE}')
    print(f'Saved: {CRITIC_SAVE}')


if __name__ == '__main__':
    train()
