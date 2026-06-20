"""
model_mg2.py

元論文アーキテクチャに即した SuperpositionNetwork。
model_mg.py からの変更点:

    model_mg.py:
        om_generated = motion_generator(h²_{t-1})   ← 前ステップ隠れ状態を入力

    model_mg2.py (本ファイル):
        om_generated = motion_generator(ov_enc)      ← 現ステップ視覚特徴量を入力

元論文 model.py SuperpositionNetworkMotionGeneration と一致:
    ov_enc = self.other_vision_encoder_module(sv)
    om = self.motion_generator_module(ov_enc)   ← ov_enc が MG と LSTM 両方に入力

効果:
    ov_enc への勾配が MG 経路からも流れ込む
    → other_enc が A-2 の視点情報を ov_enc に強く表現するよう学習する
    → decoder(ov_enc) による視点取得が成立しやすくなる

ValueEstimator は引き続き h²_{t-1} を使用する:
    Q値推測には A-2 の時系列コンテキストが必要であり、
    ov_enc (瞬時特徴量) より h²_{t-1} (蓄積状態) が適切なため。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from model import util
from model.base import WithRNNModuleBase
from model.modules import (
    FeaturePredictionModule,
    IntegrationModule,
    VisionDecoderModule,
    VisionEncoderModule,
)
from modules_mg import SuperpositionModuleWithMG    # 変更なし、再利用
from modules_mg2 import MotionGeneratorForOtherV2   # ov_enc 入力版 MG
from value_estimator_mg import ValueEstimatorMG     # 変更なし、再利用


class SuperpositionNetworkWithMG2(nn.Module, WithRNNModuleBase):
    """
    元論文アーキテクチャに即した SuperpositionNetwork (MG2 版)。

    model_mg.py (SuperpositionNetworkWithMG) との差分:
        motion_generator の入力: h²_{t-1} → ov_enc
        motion_generator のクラス: MotionGeneratorForOther → MotionGeneratorForOtherV2

    モジュール構成:
        self_vision_encoder_module  : 飯塚モデルから転送可
        other_vision_encoder_module : 飯塚モデルから転送可
        share_lns                   : 飯塚モデルから転送可
        superposition_module        : SuperpositionModuleWithMG (ランダム初期化)
        motion_generator            : MotionGeneratorForOtherV2 (ov_enc 入力)
        value_estimator             : ValueEstimatorMG (h²_{t-1} 入力)
        integration_module          : 飯塚モデルから転送可
        vision_decoder_module       : 飯塚モデルから転送可
        feature_prediction_module   : 飯塚モデルから転送可
    """

    def __init__(self, config, q_dim: int = 1, mg_hidden: int = 64):
        super().__init__()
        self.config = config
        self.q_dim = q_dim

        self.rnn_modules = []

        # ---- 飯塚互換モジュール ----
        self.share_lns = nn.ModuleList([
            nn.LayerNorm(config.vision_encoder_module.fc[i].output)
            for i in range(len(config.vision_encoder_module.fc))
        ])
        self.self_vision_encoder_module = VisionEncoderModule(
            config.vision_encoder_module, self.share_lns)
        self.other_vision_encoder_module = VisionEncoderModule(
            config.vision_encoder_module, self.share_lns)
        self.integration_module = IntegrationModule(config.integration_module)
        self.vision_decoder_module = VisionDecoderModule(
            config.vision_decoder_module,
            self.self_vision_encoder_module.get_convolved_shape())
        self.feature_prediction_module = FeaturePredictionModule(
            config.feature_prediction_module)

        # ---- 新規モジュール ----
        sp_hidden  = config.superposition_module.hidden        # 128
        enc_dim    = config.superposition_module.input.vision  # 64
        motion_dim = config.superposition_module.input.motion  # 2

        self.superposition_module = SuperpositionModuleWithMG(
            config.superposition_module, q_dim=q_dim)
        self.rnn_modules.append(self.superposition_module)

        # MG: ov_enc (64) → om_generated (2)   ← 元論文と同じ入力
        self.motion_generator = MotionGeneratorForOtherV2(
            enc_dim=enc_dim, mg_hidden=mg_hidden, motion_dim=motion_dim)
        self.rnn_modules.append(self.motion_generator)

        # ValueEstimator: h²_{t-1} (128) → q_other (1)
        self.value_estimator = ValueEstimatorMG(state_dim=sp_hidden)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: dict, q_self: torch.Tensor,
                p_mask_vision_self: float, p_mask_vision_other: float):
        """
        x['self_vision']  : (B, 3, 16, 64)
        x['self_motion']  : (B, 2)
        q_self            : (B, 1)

        ※ 'other_motion' は不要。MG が ov_enc から内部生成する。

        return:
            pred : {'self_vision': (B, 3, 16, 64)}
            ss   : h¹_t  (B, 128)
            os   : h²_t  (B, 128)
        """
        sv = x['self_vision']
        sm = x['self_motion']

        # ---- VisionEncoder ----
        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)

        # ---- Motion Generator: ov_enc → om_generated  (元論文と同じ経路) ----
        # マスク前の ov_enc を使う (元論文の実装に準拠)
        om_generated = self.motion_generator(ov_enc)

        # ---- ValueEstimator: h²_{t-1} → q_other  (時系列コンテキストを使用) ----
        oh_prev = self.superposition_module.state['other'].hidden.detach()
        q_other = self.value_estimator(oh_prev)

        # ---- マスク ----
        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        # ---- Φ_s ----
        ss, os = self.superposition_module(
            sv_enc, sm, q_self,
            ov_enc, om_generated, q_other,
        )

        # ---- Integration → VisionDecoder ----
        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )
        pred = {'self_vision': self.vision_decoder_module(so)}

        return pred, ss, os

    def predict_feature(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_prediction_module(x)

    # ------------------------------------------------------------------
    # 飯塚モデルからの重み転送
    # ------------------------------------------------------------------

    def load_iizuka_weights(self, checkpoint: dict, device):
        state_dict = checkpoint.get('model', checkpoint)
        compatible_keys = [
            'self_vision_encoder_module',
            'other_vision_encoder_module',
            'share_lns',
            'integration_module',
            'vision_decoder_module',
            'feature_prediction_module',
        ]
        own_dict = self.state_dict()
        loaded = skipped = 0
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in compatible_keys):
                if k in own_dict and own_dict[k].shape == v.shape:
                    own_dict[k] = v
                    loaded += 1
                else:
                    skipped += 1
            else:
                skipped += 1
        self.load_state_dict(own_dict)
        print(f"[load_iizuka_weights] loaded={loaded}, skipped={skipped}")
