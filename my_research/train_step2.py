import sys
sys.path.insert(0, '/work')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from model.model import SuperpositionNetworkFeaturePrediction
from rl_agent_lstm import DQNAgentLSTM, ACTIONS, N_ACTIONS
from q_estimator import QEstimator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 500
MAX_STEPS = 100
LR = 0.0001
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

def train():
    # 飯塚さんのモデルを読み込む（固定）
    model_config = load_exp_config(
        '/work/config/model/SuperpositionNetworkFeaturePrediction/default.yml'
    )
    iizuka_model = SuperpositionNetworkFeaturePrediction(model_config).to(DEVICE)
    iizuka_model.load_state_dict(
        torch.load('/work/data/result/exp1_l1/0/model/00200.pth',
                   map_location=DEVICE),
        strict=False
    )
    iizuka_model.eval()
    # 飯塚さんのモデルは学習しない
    for param in iizuka_model.parameters():
        param.requires_grad = False

    # 自己エージェント（学習済みRL・固定）
    self_agent = DQNAgentLSTM().to(DEVICE)
    self_agent.load_state_dict(
        torch.load('/work/my_research/rl_model_v5.pth')
    )
    self_agent.eval()
    for param in self_agent.parameters():
        param.requires_grad = False

    # Q値推測モジュール（これを学習させる）
    q_estimator = QEstimator().to(DEVICE)
    optimizer = optim.Adam(q_estimator.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 環境の初期化
    env_config = load_config(
        '/work/simulation/config/collect/self_random_other_stay.yml'
    )
    env = creator.create_environment(env_config.environment)
    env.init()
    env.off_display()

    print(f"学習開始 ({DEVICE})")
    print(f"Step 2：QEstimatorの学習")

    losses = []

    for episode in range(EPISODES):
        env.reset()
        v_raw, sm, om, sp, op = env.step()
        state = preprocess(v_raw)
        hidden = None

        iizuka_model.superposition_module.init_state(1)
        episode_loss = 0

        for step in range(MAX_STEPS):
            # 自己エージェントの行動
            action, action_idx, hidden = self_agent.get_action(
                state, hidden, epsilon=0.05
            )

            # 環境を1ステップ進める
            env.self_agent.p = env.self_agent.p + action
            env.self_agent.p = np.clip(env.self_agent.p, -9.5, 9.5)
            v_next_raw, sm_next, om_next, sp_next, op_next = env.step()
            next_state = preprocess(v_next_raw)

            # 飯塚さんのモデルからosを取り出す
            v_t = torch.FloatTensor(v_raw).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            sm_t = torch.FloatTensor(sm).unsqueeze(0).to(DEVICE)
            om_t = torch.zeros_like(sm_t)

            with torch.no_grad():
                sv_enc = iizuka_model.self_vision_encoder_module(v_t)
                ov_enc = iizuka_model.other_vision_encoder_module(v_t)
                ss, os = iizuka_model.superposition_module(
                    sv_enc, sm_t, ov_enc, om_t
                )

            # 実際の報酬（正解データ）
            actual_reward = get_reward_from_vision(v_next_raw)
            actual_reward_t = torch.FloatTensor([actual_reward]).to(DEVICE).squeeze()

            # QEstimatorの予測
            om_t_input = torch.FloatTensor(om).unsqueeze(0).to(DEVICE)
            predicted_q = q_estimator(os, om_t_input).squeeze()

            # 損失計算・更新
            loss = criterion(predicted_q, actual_reward_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss.item()

            v_raw = v_next_raw
            sm = sm_next
            state = next_state

        losses.append(episode_loss / MAX_STEPS)

        if episode % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            print(f"Episode {episode:4d} | 平均Loss: {avg_loss:.6f}")

    torch.save(q_estimator.state_dict(),
               '/work/my_research/q_estimator.pth')
    print("学習完了: q_estimator.pth を保存しました")

if __name__ == "__main__":
    train()
