import sys
sys.path.append('/work')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from simulation import creator
from simulation.util import load_config
from rl_agent import DQNAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 2000
MAX_STEPS = 100
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 10000
LR = 0.0001
TARGET_UPDATE = 10

# 閾値
RED_THRESHOLD = 50    # 赤ピクセルがこれ以上見えたら近い
GREEN_THRESHOLD = 50  # 緑ピクセルがこれ以上見えたら近い

def get_reward_from_vision(v_raw):
    """
    スパース報酬設計：
    赤が十分大きく見えたら+1
    緑が十分大きく見えたら-1
    それ以外は0
    """
    r = v_raw[:, :, 0]
    g = v_raw[:, :, 1]
    b = v_raw[:, :, 2]

    red_pixels   = int(((r > 0.9) & (g < 0.1) & (b < 0.1)).sum())
    green_pixels = int(((g > 0.9) & (r < 0.1)  & (b < 0.1)).sum())

    reward = 0.0
    if red_pixels > RED_THRESHOLD:
        reward += 1.0
    if green_pixels > GREEN_THRESHOLD:
        reward -= 1.0

    return reward

def preprocess(v):
    v = np.transpose(v, (2, 0, 1))
    v = torch.FloatTensor(v).unsqueeze(0).to(DEVICE)
    return v

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train():
    config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
    env = creator.create_environment(config.environment)
    env.init()
    env.off_display()

    model = DQNAgent().to(DEVICE)
    target_model = DQNAgent().to(DEVICE)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    episode_rewards = []

    print(f"学習開始 ({DEVICE})")
    print(f"報酬設計：赤に近づいたら+1、緑に近づいたら-1（スパース報酬）")
    print(f"閾値：赤>{RED_THRESHOLD}px、緑>{GREEN_THRESHOLD}px")

    for episode in range(EPISODES):
        env.reset()
        v_raw, _, _, _, _ = env.step()
        state = preprocess(v_raw)
        total_reward = 0

        for step in range(MAX_STEPS):
            action, q = model.get_action(state, epsilon)
            action_np = action.cpu().numpy()

            env.self_agent.p = env.self_agent.p + action_np
            env.self_agent.p = np.clip(env.self_agent.p, -9.5, 9.5)

            v_next_raw, _, _, _, _ = env.step()
            next_state = preprocess(v_next_raw)

            reward = get_reward_from_vision(v_next_raw)
            total_reward += reward

            done = (step == MAX_STEPS - 1)

            memory.push(state, action_np, reward, next_state, done)
            state = next_state

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states)
                next_states = torch.cat(next_states)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)

                q_values, _ = model(states)
                q_values = q_values.squeeze(1)

                with torch.no_grad():
                    next_q, _ = target_model(next_states)
                    next_q = next_q.squeeze(1)
                    target_q = rewards + GAMMA * next_q * (1 - dones)

                loss = criterion(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        if episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode:4d} | 平均報酬: {avg_reward:.2f} | epsilon: {epsilon:.3f}")

    torch.save(model.state_dict(), '/work/my_research/rl_model_v3.pth')
    print("学習完了: rl_model_v3.pth を保存しました")
    return episode_rewards

if __name__ == "__main__":
    train()
