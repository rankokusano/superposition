"""
v2 RL agent training: A-1 learns red-preference + green-aversion via SAC.

Reward design:
    Red visible (>50 px): +1.0   (same as v1)
    Green visible (>50 px): -2.0 (stronger than v1 to widen Q-value range)

Saves to:
    /work/my_research/preference_inference/data/model/v2_rl_actor.pth
    /work/my_research/preference_inference/data/model/v2_rl_critic.pth

Run inside Docker with xvfb:
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \\
        python my_research/preference_inference/simulation/train_rl_v2.py
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
from my_research.rl_agent_sac import ActorLSTM, CriticLSTM

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPISODES = 2000
MAX_STEPS = 200
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001
LR_ALPHA = 0.0001
BATCH_SIZE = 32
MEMORY_SIZE = 50000
TARGET_ENTROPY = -2.0
RED_THRESHOLD = 50
GREEN_THRESHOLD = 50

SAVE_DIR = os.path.join('/work', 'my_research', 'preference_inference', 'data', 'model')
ACTOR_SAVE = os.path.join(SAVE_DIR, 'v2_rl_actor.pth')
CRITIC_SAVE = os.path.join(SAVE_DIR, 'v2_rl_critic.pth')


def get_reward_v2(vision_np):
    """Compute reward from float [0,1] HWC vision array.

    Red landmark:   +1.0 (positive preference)
    Green landmark: -2.0 (stronger aversion than v1's -1.0)
    """
    r, g, b = vision_np[:, :, 0], vision_np[:, :, 1], vision_np[:, :, 2]
    red_pixels   = int(((r > 0.9) & (g < 0.1) & (b < 0.1)).sum())
    green_pixels = int(((g > 0.9) & (r < 0.1) & (b < 0.1)).sum())
    reward = 0.0
    if red_pixels > RED_THRESHOLD:
        reward += 1.0
    if green_pixels > GREEN_THRESHOLD:
        reward -= 2.0
    return reward


def vision_to_tensor(vision_np):
    """Float32 [0,1] HWC → 1x3xHxW tensor (no normalization needed)."""
    v = vision_np.astype(np.float32)
    v = np.transpose(v, (2, 0, 1))
    return torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(DEVICE)


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

    env = creator.create_environment()
    env.off_display()
    env.init()
    env.set_camera('self')
    world = env.world

    actor          = ActorLSTM().to(DEVICE)
    critic1        = CriticLSTM().to(DEVICE)
    critic2        = CriticLSTM().to(DEVICE)
    critic1_target = CriticLSTM().to(DEVICE)
    critic2_target = CriticLSTM().to(DEVICE)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha = log_alpha.exp()

    actor_opt   = optim.Adam(actor.parameters(),    lr=LR_ACTOR)
    critic1_opt = optim.Adam(critic1.parameters(),  lr=LR_CRITIC)
    critic2_opt = optim.Adam(critic2.parameters(),  lr=LR_CRITIC)
    alpha_opt   = optim.Adam([log_alpha],            lr=LR_ALPHA)

    memory = ReplayBuffer(MEMORY_SIZE)
    episode_rewards = []
    bound = world.get_boundary()

    print(f'Device: {DEVICE}')
    print(f'Training v2 RL agent: red +1, green -2 | {EPISODES} ep × {MAX_STEPS} steps')

    for episode in range(EPISODES):
        env.reset()
        env.set_camera('self')
        sv = env.capture()
        state = vision_to_tensor(sv)
        total_reward = 0.0

        for step in range(MAX_STEPS):
            with torch.no_grad():
                action_t, _, _ = actor.sample(state, hidden=None)
            action_np = action_t.squeeze(0).cpu().numpy()

            sp = env.self_agent.get_position().copy()
            new_sp = np.clip(sp + action_np,
                             [bound[0][0], bound[1][0]],
                             [bound[0][1], bound[1][1]])
            env.set_agent_pos(new_sp, env.other_agent.get_position().copy())

            sv_next = env.capture()
            reward = get_reward_v2(sv_next)
            total_reward += reward

            next_state = vision_to_tensor(sv_next)
            done = (step == MAX_STEPS - 1)
            memory.push(state, action_t, reward, next_state, done)
            state = next_state

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states      = torch.cat(states)
                actions     = torch.cat(actions)
                rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
                next_states = torch.cat(next_states)
                dones       = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

                with torch.no_grad():
                    na, nlp, _ = actor.sample(next_states)
                    q1n, _ = critic1_target(next_states, na)
                    q2n, _ = critic2_target(next_states, na)
                    target_q = rewards + GAMMA * (1 - dones) * (
                        torch.min(q1n, q2n) - alpha * nlp)

                q1, _ = critic1(states, actions)
                q2, _ = critic2(states, actions)
                loss_c1 = nn.MSELoss()(q1, target_q)
                loss_c2 = nn.MSELoss()(q2, target_q)

                critic1_opt.zero_grad(); loss_c1.backward(); critic1_opt.step()
                critic2_opt.zero_grad(); loss_c2.backward(); critic2_opt.step()

                na2, lp2, _ = actor.sample(states)
                q1a, _ = critic1(states, na2)
                q2a, _ = critic2(states, na2)
                loss_a = (alpha * lp2 - torch.min(q1a, q2a)).mean()

                actor_opt.zero_grad(); loss_a.backward(); actor_opt.step()

                loss_alpha = -(log_alpha * (lp2 + TARGET_ENTROPY).detach()).mean()
                alpha_opt.zero_grad(); loss_alpha.backward(); alpha_opt.step()
                alpha = log_alpha.exp()

                soft_update(critic1_target, critic1, TAU)
                soft_update(critic2_target, critic2, TAU)

        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f'  ep {episode:5d} | avg_reward(100ep): {avg:.3f}')

    torch.save(actor.state_dict(), ACTOR_SAVE)
    torch.save(critic1.state_dict(), CRITIC_SAVE)
    print(f'Saved: {ACTOR_SAVE}')
    print(f'Saved: {CRITIC_SAVE}')


if __name__ == '__main__':
    train()
