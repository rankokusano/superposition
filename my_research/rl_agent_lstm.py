import torch
import torch.nn as nn
import numpy as np

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

class DQNAgentLSTM(nn.Module):
    def __init__(self, hidden_dim=128):
        super(DQNAgentLSTM, self).__init__()

        # CNN：画像から特徴量を抽出
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.cnn_out_dim = 64 * 4 * 16

        # CNN出力を小さくする全結合層
        self.cnn_fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim, 256),
            nn.ReLU(),
        )

        # LSTM：過去の状態を記憶
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Q値を出力
        self.q_head = nn.Linear(hidden_dim, N_ACTIONS)

        self.hidden_dim = hidden_dim

    def forward(self, v, hidden=None):
        """
        v: 視覚入力 [batch, 3, 16, 64]
        hidden: LSTMの隠れ状態（Noneなら初期化）
        """
        batch_size = v.size(0)

        # CNN
        x = self.cnn(v)
        x = x.view(batch_size, -1)
        x = self.cnn_fc(x)

        # LSTMへの入力は[batch, seq=1, features]
        x = x.unsqueeze(1)

        # LSTM
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(1)

        # Q値
        q_values = self.q_head(x)

        return q_values, hidden

    def get_action(self, v, hidden=None, epsilon=0.1):
        if np.random.random() < epsilon:
            action_idx = np.random.randint(N_ACTIONS)
            with torch.no_grad():
                _, new_hidden = self.forward(v, hidden)
        else:
            with torch.no_grad():
                q_values, new_hidden = self.forward(v, hidden)
            action_idx = q_values.argmax().item()

        action = ACTIONS[action_idx]
        return action, action_idx, new_hidden
