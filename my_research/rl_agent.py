import torch
import torch.nn as nn
import numpy as np

class DQNAgent(nn.Module):
    """
    教授指示「1.強化学習モジュール」
    入力：v_t（16×64×3の画像）
    出力：Q値（実数）と行動a（2次元ベクトル、m_tと同じフォーマット）
    """
    def __init__(self):
        super(DQNAgent, self).__init__()

        # 画像を特徴量に変換するCNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # 16×64 → stride2で8×32 → stride2で4×16
        # 64チャンネル × 4 × 16 = 4096
        self.cnn_out_dim = 64 * 4 * 16

        # 特徴量 → Q値と行動を出力する全結合層
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Q値の出力（実数1つ）
        self.q_head = nn.Linear(128, 1)

        # 行動の出力（2次元ベクトル、m_tと同じフォーマット）
        self.action_head = nn.Sequential(
            nn.Linear(128, 2),
            nn.Tanh()  # -1〜1に収める（速度v=1に対応）
        )

    def forward(self, v):
        """
        v: 視覚入力 [batch, 3, 16, 64]
        戻り値：
            q: Q値 [batch, 1]
            a: 行動 [batch, 2]（m_tと同じフォーマット）
        """
        x = self.cnn(v)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        q = self.q_head(x)
        a = self.action_head(x)
        return q, a

    def get_action(self, v, epsilon=0.1):
        """
        ε-greedy法で行動を選択
        epsilon: ランダム行動の確率
        """
        if np.random.random() < epsilon:
            # ランダム行動（探索）
            a = np.random.uniform(-1, 1, size=2)
            return torch.FloatTensor(a), None
        else:
            # モデルの出力に従う（活用）
            with torch.no_grad():
                q, a = self.forward(v)
            return a.squeeze(0), q.squeeze(0)
