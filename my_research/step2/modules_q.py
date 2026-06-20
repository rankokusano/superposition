"""
modules_q.py

飯塚ネットワークのSuperpositionModule (Φ_s) をQ値入力対応に拡張したモジュール。
元のmodules.pyをコピー・修正。

変更点：
    SuperpositionModule → SuperpositionModuleWithQ
        - LSTMの入力次元を +1（Q値）
        - forward()がq_self, q_otherを追加引数として受け取る
"""

from collections import namedtuple

import torch
import torch.nn as nn

from model.base import RNNBase

LSTMState = namedtuple('LSTMState', ('hidden', 'cell'))


class SuperpositionModuleWithQ(nn.Module, RNNBase):
    """
    Φ_s (SuperpositionModule) のQ値入力拡張版。

    元のLSTM入力次元: vision(64) + motion(2) = 66
    本モジュール:      vision(64) + motion(2) + Q(q_dim) = 66 + q_dim

    Process-1 (A-1): q_self = A-1のCriticが出力するQ値（固定・外部から受け取る）
    Process-2 (A-2): q_other = ValueEstimatorが推測したQ値（学習対象）

    両プロセスで同一のLSTMパラメータを共有する点は元実装と同じ。
    """

    def __init__(self, config, q_dim=1):
        super().__init__()
        self.config = config
        self.q_dim = q_dim
        # vision + motion + Q
        lstm_input_size = config.input.motion + config.input.vision + q_dim
        self.lstm = nn.LSTMCell(lstm_input_size, config.hidden)

    def init_state(self, batch_size):
        device = next(self.lstm.parameters()).device
        zeros = lambda: torch.zeros((batch_size, self.config.hidden), device=device)
        self.state = {
            'self':  LSTMState(zeros(), zeros()),
            'other': LSTMState(zeros(), zeros()),
        }

    def forward(self, sv, sm, q_self, ov, om, q_other):
        """
        sv:     自己視覚エンコーディング  (B, vision_dim=64)
        sm:     自己の運動               (B, motion_dim=2)
        q_self: Process-1 の Q値         (B, q_dim=1)  ← A-1 Critic の出力
        ov:     他者視覚エンコーディング  (B, vision_dim=64)
        om:     他者の運動               (B, motion_dim=2)
        q_other:Process-2 の推測Q値      (B, q_dim=1)  ← ValueEstimator の出力
        """
        sh, sc = self.state['self']
        oh, oc = self.state['other']

        sx = torch.cat([sv, sm, q_self], dim=1)   # (B, 64+2+1=67)
        ox = torch.cat([ov, om, q_other], dim=1)  # (B, 64+2+1=67)

        next_sh, next_sc = self.lstm(sx, (sh, sc))
        next_oh, next_oc = self.lstm(ox, (oh, oc))

        self.state['self']  = LSTMState(next_sh, next_sc)
        self.state['other'] = LSTMState(next_oh, next_oc)

        return next_sh, next_oh  # h¹_t, h²_t
