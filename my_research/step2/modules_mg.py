"""
modules_mg.py

Motion Generator 版 Superposition モジュール群。
modules_q.py を保持したまま、新設計として独立したファイル。

変更の動機:
    modules_q.py では Process-2 の om 入力に GreenFollower の真の行動ベクトルを
    直接渡していた。これにより LSTM は視覚推論をバイパスして om を記憶するだけで
    最小損失を達成できるため、A-2 の視覚世界構築・意図推測のインセンティブが消える。

設計:
    MotionGeneratorForOther
        h²_{t-1} → MG-LSTM → om_generated
        A-1 が「A-2 はこう動くはず」と内部生成する運動表現。

    SuperpositionModuleWithMG
        Process-1: LSTM(sv_enc, sm,           q_self)  → h¹
        Process-2: LSTM(ov_enc, om_generated, q_other) → h²
        om_generated は MotionGeneratorForOther が前ステップ h² から生成する。
        GreenFollower の真の行動は一切参照しない。

野口ら論文との対応:
    スライド 29-33 の Motion Generator(MG) に相当する。
    論文では MG の隠れ状態が CW/CCW/Stopping のアトラクタを形成した。
    本設計では「緑へ向かう意図」のアトラクタが h² に形成されることを期待する。
"""

from collections import namedtuple

import torch
import torch.nn as nn

from model.base import RNNBase

LSTMState = namedtuple('LSTMState', ('hidden', 'cell'))


class MotionGeneratorForOther(nn.Module, RNNBase):
    """
    A-2 の運動を h²_{t-1} から内部生成するモジュール（野口論文の MG に相当）。

    入力: h²_{t-1}  (B, sp_hidden=128)
    出力: om_generated  (B, motion_dim=2)

    MG は自身の LSTM 状態を持ち、Process-2 の Φ_s とは独立して進化する。
    これにより「A-1 が A-2 の動きをどう想像するか」が MG 内部に蓄積される。

    Args:
        sp_hidden  : Process-2 の隠れ状態次元 (= Φ_s の hidden, 通常 128)
        mg_hidden  : MG 自身の LSTM 隠れ次元 (通常 64)
        motion_dim : 出力運動ベクトルの次元 (通常 2)
    """

    def __init__(self, sp_hidden: int = 128, mg_hidden: int = 64, motion_dim: int = 2):
        super().__init__()
        self.mg_hidden = mg_hidden

        self.lstm = nn.LSTMCell(sp_hidden, mg_hidden)
        self.fc = nn.Linear(mg_hidden, motion_dim)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def init_state(self, batch_size: int):
        device = next(self.lstm.parameters()).device
        zeros = lambda: torch.zeros((batch_size, self.mg_hidden), device=device)
        self.state = {'motion_generator': LSTMState(zeros(), zeros())}

    def forward(self, h_other_prev: torch.Tensor) -> torch.Tensor:
        """
        h_other_prev : h²_{t-1}  (B, sp_hidden)
        return       : om_generated  (B, motion_dim)  ∈ (-1, 1)
        """
        h, c = self.state['motion_generator']
        next_h, next_c = self.lstm(h_other_prev, (h, c))
        self.state['motion_generator'] = LSTMState(next_h, next_c)
        return torch.tanh(self.fc(next_h))


class SuperpositionModuleWithMG(nn.Module, RNNBase):
    """
    Q値入力 + Motion Generator 版 Φ_s。

    元の SuperpositionModuleWithQ (modules_q.py) との差分:
        om 引数が om_true (外部) から om_generated (MG 出力) に変わる。
        引数シグネチャ上は同じ om だが、呼び出し元が何を渡すかが変わる。

    LSTM 入力次元:
        vision(64) + motion(2) + Q(q_dim=1) = 67  ← modules_q.py と同じ

    両プロセスで同一の LSTM 重みを共有（原論文・飯塚実装の方針を継承）。
    """

    def __init__(self, config, q_dim: int = 1):
        super().__init__()
        self.config = config
        self.q_dim = q_dim
        lstm_input_size = config.input.motion + config.input.vision + q_dim
        self.lstm = nn.LSTMCell(lstm_input_size, config.hidden)

    def init_state(self, batch_size: int):
        device = next(self.lstm.parameters()).device
        zeros = lambda: torch.zeros((batch_size, self.config.hidden), device=device)
        self.state = {
            'self':  LSTMState(zeros(), zeros()),
            'other': LSTMState(zeros(), zeros()),
        }

    def forward(
        self,
        sv: torch.Tensor,           # 自己視覚エンコード (B, 64)
        sm: torch.Tensor,           # 自己運動 (B, 2)
        q_self: torch.Tensor,       # A-1 の Q値 (B, 1)
        ov: torch.Tensor,           # 他者視覚エンコード (B, 64)
        om_generated: torch.Tensor, # MG 生成運動 (B, 2)  ← 真の om ではない
        q_other: torch.Tensor,      # 推測 Q値 (B, 1)
    ):
        sh, sc = self.state['self']
        oh, oc = self.state['other']

        sx = torch.cat([sv, sm,           q_self],  dim=1)  # (B, 67)
        ox = torch.cat([ov, om_generated, q_other], dim=1)  # (B, 67)

        next_sh, next_sc = self.lstm(sx, (sh, sc))
        next_oh, next_oc = self.lstm(ox, (oh, oc))

        self.state['self']  = LSTMState(next_sh, next_sc)
        self.state['other'] = LSTMState(next_oh, next_oc)

        return next_sh, next_oh  # h¹_t, h²_t
