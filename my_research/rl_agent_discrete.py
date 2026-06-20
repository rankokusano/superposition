import torch
import torch.nn as nn
import numpy as np

# 9方向の離散行動
ACTIONS = [
    np.array([-1.0,  1.0]),  # 0: 左上（赤方向）
    np.array([ 0.0,  1.0]),  # 1: 上
    np.array([ 1.0,  1.0]),  # 2: 右上
    np.array([-1.0,  0.0]),  # 3: 左
    np.array([ 0.0,  0.0]),  # 4: 停止
    np.array([ 1.0,  0.0]),  # 5: 右
    np.array([-1.0, -1.0]),  # 6: 左下（緑方向）
    np.array([ 0.0, -1.0]),  # 7: 下
    np.array([ 1.0, -1.0]),  # 8: 右下
]
N_ACTIONS = len(ACTIONS)

class DQNAgentDiscrete(nn.Module):
    def __init__(self):
        super(DQNAgentDiscrete, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.cnn_out_dim = 64 * 4 * 16

        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 9つの行動それぞれのQ値を出力
        self.q_head = nn.Linear(128, N_ACTIONS)

    def forward(self, v):
        x = self.cnn(v)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        q_values = self.q_head(x)
        return q_values

    def get_action(self, v, epsilon=0.1):
        if np.random.random() < epsilon:
            # ランダム行動
            action_idx = np.random.randint(N_ACTIONS)
        else:
            with torch.no_grad():
                q_values = self.forward(v)
            action_idx = q_values.argmax().item()

        action = ACTIONS[action_idx]
        return action, action_idx
