"""
value_estimator.py

Q(s,a) モジュール — Process-2 (A-2) のQ値を推測する全結合ネットワーク。

設計仕様（研究概要より）:
    入力: s_t (h²_t : 128次元) + a (m_t : 2次元) = 130次元
    出力: Q値 (実数1つ)
    構造: 全結合3層

NOTE: 前ステップの隠れ状態 h²_{t-1} を入力に使うため、
      循環参照（h²_t → Q → h²_t）が生じない。
"""

import torch
import torch.nn as nn


class ValueEstimator(nn.Module):
    """
    Process-2 (A-2) のQ値を推測するモジュール。

    入力:
        state  : h²_{t-1}  (B, state_dim=128)  — Φ_s の前ステップ隠れ状態
        action : om_t      (B, action_dim=2)   — A-2 の運動ベクトル
    出力:
        q_est  : 推測Q値   (B, 1)
    """

    def __init__(self, state_dim: int = 128, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state  : (B, 128)
        action : (B, 2)
        return : (B, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
