import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class CNNEncoder(nn.Module):
    """画像から特徴量を抽出するCNN"""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 16, 256),
            nn.ReLU(),
        )

    def forward(self, v):
        x = self.cnn(v)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ActorLSTM(nn.Module):
    """
    行動を決めるネットワーク（Actor）
    CNN + LSTM → 平均・標準偏差を出力
    → 連続行動をサンプリング
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(256, hidden_dim, batch_first=True)
        self.mean = nn.Linear(hidden_dim, 2)
        self.log_std = nn.Linear(hidden_dim, 2)
        self.hidden_dim = hidden_dim

    def forward(self, v, hidden=None):
        batch_size = v.size(0)
        feat = self.encoder(v).unsqueeze(1)
        lstm_out, hidden = self.lstm(feat, hidden)
        lstm_out = lstm_out.squeeze(1)
        mean = self.mean(lstm_out)
        log_std = self.log_std(lstm_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, hidden

    def sample(self, v, hidden=None):
        mean, log_std, hidden = self.forward(v, hidden)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, hidden

class CriticLSTM(nn.Module):
    """
    Q値を評価するネットワーク（Critic）
    CNN + LSTM + 行動 → Q値
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = CNNEncoder()
        self.lstm = nn.LSTM(256, hidden_dim, batch_first=True)
        self.q = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, v, action, hidden=None):
        feat = self.encoder(v).unsqueeze(1)
        lstm_out, hidden = self.lstm(feat, hidden)
        lstm_out = lstm_out.squeeze(1)
        x = torch.cat([lstm_out, action], dim=-1)
        q = self.q(x)
        return q, hidden
