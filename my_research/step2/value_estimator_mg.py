"""
value_estimator_mg.py

Motion Generator 版 ValueEstimator。
value_estimator.py を保持したまま、新設計として独立したファイル。

変更の動機:
    value_estimator.py は h²_{t-1} + om_true を入力に使っていた。
    om_true = GreenFollower.get_action() を入力することは、
    「A-1 が A-2 の真の行動を知っている」ことを前提とする。
    これは model_q.py の問題（Process-2 への om_true 直接入力）と同根であり、
    A-2 の価値観推測の検証を妨げる。

設計:
    h²_{t-1} のみを入力とし、「内部状態だけから A-2 の価値を推測できるか」を検証する。
    om_generated (MG 出力) も入力しない理由:
        MG 出力は A-1 の想像であって A-2 の行動の ground truth ではない。
        ValueEstimator には「h² が価値情報を本当に含んでいるか」を純粋に問いたい。
"""

import torch
import torch.nn as nn


class ValueEstimatorMG(nn.Module):
    """
    Process-2 の Q値を h²_{t-1} のみから推測するモジュール。

    value_estimator.py (ValueEstimator) との差分:
        入力から om (action) を除去。
        state_dim のみを入力とする 3 層 MLP。

    Args:
        state_dim  : h² の次元 (= Φ_s の hidden, 通常 128)
        hidden_dim : 中間層次元 (通常 64)

    入力: h²_{t-1}  (B, state_dim)
    出力: q_est     (B, 1)
    """

    def __init__(self, state_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state  : h²_{t-1}  (B, state_dim)
        return : q_est      (B, 1)
        """
        return self.net(state)
