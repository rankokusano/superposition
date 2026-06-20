"""
value_estimator_v2.py

ValueEstimator 第2版。
step2/value_estimator_mg.py (V(h²)) から設計を変更。

【変更の動機】
    mg5 の ValueEstimatorMG は h²_{t-1} を入力としていた。
    これは以下の問題を生む：
        (1) 循環依存: h² → q_other → SM → h²
        (2) 学習初期に未学習の h² からランダムな q_other が SM に流れ込み、
            h² の学習（位置追跡）を乱す

【新設計: Q(s, a) 対称型】
    Process-1 の CriticLSTM と対称な入力設計にする。

        Process-1: q_self  = CriticLSTM(v_t, m_t)
                   入力: (視覚画像, 行動)

        Process-2: q_other = ValueEstimatorV2(ov_enc, om_generated)
                   入力: (encoder-2 の出力, MG が推定した行動)

    ov_enc       : encoder-2 が A-1 の視覚を処理した特徴量  (64-dim)
                   → A-1 から見た A-2 の状態を表す
    om_generated : MG が推定した A-2 の運動                  (2-dim)
                   → A-1 が想像した A-2 の行動を表す

【利点】
    - 循環依存が解消される（ov_enc は encoder 出力で h² に依存しない）
    - 学習初期から安定した勾配が ValueEstimator に流れる
    - Q(s,a) 型なので、任意の位置・行動でクエリ可能
      → ヒートマップ作成時に「A-2 を位置 X に置いて q_other を計算」が直接できる
"""

import torch
import torch.nn as nn


class ValueEstimatorV2(nn.Module):
    """
    Process-2 の Q値を (ov_enc, om_generated) から推測するモジュール。

    CriticLSTM(v_t, action) → q_self と対称な設計。
    （CriticLSTM は視覚+行動 → Q値。本モジュールは ov_enc+om_gen → Q値）

    Args:
        enc_dim    : ov_enc の次元 (通常 64)
        motion_dim : om_generated の次元 (通常 2)
        hidden_dim : 中間層次元 (通常 64)

    入力:
        ov_enc       : (B, enc_dim)     encoder-2 の出力（マスク前）
        om_generated : (B, motion_dim)  MG の出力

    出力:
        q_est : (B, 1)  A-2 の推定 Q 値
    """

    def __init__(self, enc_dim: int = 64, motion_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim + motion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ov_enc: torch.Tensor, om_generated: torch.Tensor) -> torch.Tensor:
        """
        ov_enc       : (B, enc_dim)
        om_generated : (B, motion_dim)
        return       : (B, 1)
        """
        x = torch.cat([ov_enc, om_generated], dim=-1)
        return self.net(x)
