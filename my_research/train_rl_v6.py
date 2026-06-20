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
from rl_agent_sac import ActorLSTM, CriticLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 1000
MAX_STEPS = 100
GAMMA = 0.99
TAU = 0.005           # ターゲットネットワークの更新率
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001
LR_ALPHA = 0.0001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_ENTROPY = -2.0  # 目標エントロピー（行動次元数のマイナス）
RED_THRESHOLD = 50
GREEN_THRESHOLD = 50

def get_reward_from_vision(v_raw):
    r = v_raw[:, :, 0]
    g = v_raw[:, :, 1]
    b = v_raw[:, :, 2]
    red_pixels   = int(((r > 0.9) & (g < 0.1) & (b < 0.1)).sum())
    green_pixels = int(((g > 0.9) & (r < 0.1) & (b < 0.1)).sum())
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

def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

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
    config = load_config(
        '/work/simulation/config/collect/self_random_other_stay.yml'
    )
    env = creator.create_environment(config.environment)
    env.init()
    env.off_display()

    # Actor・Critic・ターゲットCriticを初期化
    actor = ActorLSTM().to(DEVICE)
    critic1 = CriticLSTM().to(DEVICE)
    critic2 = CriticLSTM().to(DEVICE)
    critic1_target = CriticLSTM().to(DEVICE)
    critic2_target = CriticLSTM().to(DEVICE)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    # 温度パラメータα（探索と活用のバランス）
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha = log_alpha.exp()

    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=LR_CRITIC)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=LR_CRITIC)
    alpha_optimizer = optim.Adam([log_alpha], lr=LR_ALPHA)

    memory = ReplayBuffer(MEMORY_SIZE)
    episode_rewards = []

    print(f"学習開始 ({DEVICE})")
    print(f"アルゴリズム：SAC（連続行動・CNN+LSTM）")
    print(f"報酬設計：赤>50px→+1、緑>50px→-1")

    for episode in range(EPISODES):
        env.reset()
        v_raw, _, _, _, _ = env.step()
        state = preprocess(v_raw)
        hidden = None
        total_reward = 0

        for step in range(MAX_STEPS):
            # 行動をサンプリング
            with torch.no_grad():
                action, _, hidden = actor.sample(state, hidden)
            action_np = action.squeeze(0).cpu().numpy()

            # 環境を進める
            env.self_agent.p = env.self_agent.p + action_np
            env.self_agent.p = np.clip(env.self_agent.p, -9.5, 9.5)
            v_next_raw, _, _, _, _ = env.step()
            next_state = preprocess(v_next_raw)

            reward = get_reward_from_vision(v_next_raw)
            total_reward += reward
            done = (step == MAX_STEPS - 1)

            memory.push(state, action, reward, next_state, done)
            state = next_state

            # 学習
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states)
                actions = torch.cat(actions)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
                next_states = torch.cat(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(DEVICE)

                with torch.no_grad():
                    next_actions, next_log_probs, _ = actor.sample(next_states)
                    q1_next, _ = critic1_target(next_states, next_actions)
                    q2_next, _ = critic2_target(next_states, next_actions)
                    q_next = torch.min(q1_next, q2_next)
                    target_q = rewards + GAMMA * (1 - dones) * (
                        q_next - alpha * next_log_probs
                    )

                # Criticの更新
                q1, _ = critic1(states, actions)
                q2, _ = critic2(states, actions)
                critic1_loss = nn.MSELoss()(q1, target_q)
                critic2_loss = nn.MSELoss()(q2, target_q)

                critic1_optimizer.zero_grad()
                critic1_loss.backward()
                critic1_optimizer.step()

                critic2_optimizer.zero_grad()
                critic2_loss.backward()
                critic2_optimizer.step()

                # Actorの更新
                new_actions, log_probs, _ = actor.sample(states)
                q1_new, _ = critic1(states, new_actions)
                q2_new, _ = critic2(states, new_actions)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * log_probs - q_new).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # αの更新
                alpha_loss = -(
                    log_alpha * (log_probs + TARGET_ENTROPY).detach()
                ).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp()

                # ターゲットネットワークの更新
                soft_update(critic1_target, critic1, TAU)
                soft_update(critic2_target, critic2, TAU)

        episode_rewards.append(total_reward)

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode:4d} | 平均報酬: {avg_reward:.2f}")

    torch.save(actor.state_dict(), '/work/my_research/rl_model_v6_actor.pth')
    torch.save(critic1.state_dict(), '/work/my_research/rl_model_v6_critic.pth')
    print("学習完了: rl_model_v6_actor.pth を保存しました")

if __name__ == "__main__":
    train()
