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

# ===== 設定 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 500          # 学習エピソード数
MAX_STEPS = 100         # 1エピソードの最大ステップ数
GAMMA = 0.99            # 割引率（将来の報酬をどれくらい重視するか）
EPSILON_START = 1.0     # 最初はランダム行動
EPSILON_END = 0.05      # 最終的なランダム行動率
EPSILON_DECAY = 0.995   # epsilonの減衰率
BATCH_SIZE = 32         # 学習バッチサイズ
MEMORY_SIZE = 10000     # 経験を貯めるバッファサイズ
LR = 0.0001             # 学習率
TARGET_UPDATE = 10      # ターゲットネットワークの更新頻度

# ===== 報酬関数 =====
def get_reward(env):
    """
    赤いオブジェクト（座標(10,10)の角）への距離から報酬を計算
    近いほど報酬が高い
    報酬 = 1 / (距離 + 1)
    """
    self_pos, _ = env.get_agent_pos()
    red_pos = np.array([9.0, 9.0])  # 赤オブジェクトの座標（右上の角）
    dist = np.linalg.norm(self_pos - red_pos)
    reward = 1.0 / (dist + 1.0)
    return reward

# ===== 画像の前処理 =====
def preprocess(v):
    """
    環境から取得した画像をモデルの入力形式に変換
    v: numpy配列 [16, 64, 3]
    → tensor [1, 3, 16, 64]（0〜1に正規化）
    """
    v = v.astype(np.float32) / 255.0
    v = np.transpose(v, (2, 0, 1))  # [H,W,C] → [C,H,W]
    v = torch.FloatTensor(v).unsqueeze(0).to(DEVICE)
    return v

# ===== 経験リプレイバッファ =====
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ===== メイン学習 =====
def train():
    # 環境の初期化
    config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
    env = creator.create_environment(config.environment)
    env.init()
    env.off_display()

    # モデルの初期化
    model = DQNAgent().to(DEVICE)
    target_model = DQNAgent().to(DEVICE)  # ターゲットネットワーク
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    episode_rewards = []

    print(f"学習開始 ({DEVICE})")
    print(f"エピソード数: {EPISODES}, 最大ステップ: {MAX_STEPS}")

    for episode in range(EPISODES):
        env.reset()
        v, _, _, _, _ = env.step()
        state = preprocess(v)
        total_reward = 0

        for step in range(MAX_STEPS):
            # 行動を選択
            action, q = model.get_action(state, epsilon)
            action_np = action.cpu().numpy()

            # 環境を1ステップ進める
            # 自分エージェントの動きを上書き
            env.self_agent.p = env.self_agent.p + action_np
            env.self_agent.p = np.clip(env.self_agent.p, -9.5, 9.5)

            v_next, _, _, _, _ = env.step()
            next_state = preprocess(v_next)

            # 報酬を計算
            reward = get_reward(env)
            total_reward += reward

            done = (step == MAX_STEPS - 1)

            # 経験をバッファに保存
            memory.push(state, action_np, reward, next_state, done)
            state = next_state

            # バッファが十分溜まったら学習
            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.cat(states)
                next_states = torch.cat(next_states)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)

                # 現在のQ値
                q_values, _ = model(states)
                q_values = q_values.squeeze(1)

                # ターゲットQ値（ベルマン方程式）
                with torch.no_grad():
                    next_q, _ = target_model(next_states)
                    next_q = next_q.squeeze(1)
                    target_q = rewards + GAMMA * next_q * (1 - dones)

                # 損失計算・更新
                loss = criterion(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # epsilonを減衰
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        episode_rewards.append(total_reward)

        # ターゲットネットワークを定期更新
        if episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode:4d} | 平均報酬: {avg_reward:.4f} | epsilon: {epsilon:.3f}")

    # モデルを保存
    torch.save(model.state_dict(), '/work/my_research/rl_model.pth')
    print("学習完了: rl_model.pth を保存しました")
    return episode_rewards

if __name__ == "__main__":
    rewards = train()
