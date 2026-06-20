"""
modules_mg2.py

元論文アーキテクチャに即した Motion Generator モジュール。
modules_mg.py からの変更点:

    MotionGeneratorForOther  (modules_mg.py)
        入力: h²_{t-1}  (sp_hidden=128)  ← 前ステップ隠れ状態

    MotionGeneratorForOtherV2  (本ファイル)
        入力: ov_enc    (enc_dim=64)     ← 現ステップの他者視覚特徴量

元論文 model.py (SuperpositionNetworkMotionGeneration, line 127) と一致:
    om = self.motion_generator_module(ov_enc)

これにより ov_enc への勾配経路が2本になる:
    ① LSTM 入力として  (既存)
    ② MG 入力として    (追加)

→ other_enc が A-2 の視点情報を ov_enc に強く表現するよう誘導される
→ decoder(ov_enc) による視点取得が成立しやすくなる

SuperpositionModuleWithMG は modules_mg.py と同一のため再利用する。
"""

from collections import namedtuple

import torch
import torch.nn as nn

from model.base import RNNBase

LSTMState = namedtuple('LSTMState', ('hidden', 'cell'))


class MotionGeneratorForOtherV2(nn.Module, RNNBase):
    """
    ov_enc (他者視覚特徴量) から A-2 の運動を内部生成するモジュール。

    元論文 SuperpositionNetworkMotionGeneration の
        motion_generator_module(ov_enc)
    と同じ入力形式。

    MotionGeneratorForOther (modules_mg.py) との差分:
        入力次元: sp_hidden(128) → enc_dim(64)
        入力変数: h²_{t-1}       → ov_enc

    Args:
        enc_dim    : ov_enc の次元 (VisionEncoder 出力, 通常 64)
        mg_hidden  : MG 自身の LSTM 隠れ次元 (通常 64)
        motion_dim : 出力運動ベクトルの次元 (通常 2)
    """

    def __init__(self, enc_dim: int = 64, mg_hidden: int = 64, motion_dim: int = 2):
        super().__init__()
        self.mg_hidden = mg_hidden

        self.lstm = nn.LSTMCell(enc_dim, mg_hidden)
        self.fc = nn.Linear(mg_hidden, motion_dim)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def init_state(self, batch_size: int):
        device = next(self.lstm.parameters()).device
        zeros = lambda: torch.zeros((batch_size, self.mg_hidden), device=device)
        self.state = {'motion_generator': LSTMState(zeros(), zeros())}

    def forward(self, ov_enc: torch.Tensor) -> torch.Tensor:
        """
        ov_enc : 他者視覚エンコード  (B, enc_dim=64)
        return : om_generated        (B, motion_dim=2)  ∈ (-1, 1)
        """
        h, c = self.state['motion_generator']
        next_h, next_c = self.lstm(ov_enc, (h, c))
        self.state['motion_generator'] = LSTMState(next_h, next_c)
        return torch.tanh(self.fc(next_h))
