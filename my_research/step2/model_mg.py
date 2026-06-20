"""
model_mg.py

Motion Generator 版 SuperpositionNetwork。
model_q.py を保持したまま、新設計として独立したファイル。

model_q.py (SuperpositionNetworkWithQ) からの変更点:
    1. other_motion (om_true) を Process-2 LSTM に直接入力しない
    2. MotionGeneratorForOther が h²_{t-1} から om_generated を生成して Process-2 に入力
    3. ValueEstimatorMG が h²_{t-1} のみから q_other を推測 (om 依存なし)
    4. forward() の入力辞書に 'other_motion' は含めない

モジュール構成:
    self_vision_encoder_module  : 飯塚モデルから転送可
    other_vision_encoder_module : 飯塚モデルから転送可
    share_lns                   : 飯塚モデルから転送可
    superposition_module        : SuperpositionModuleWithMG  (新規 LSTM, ランダム初期化)
    motion_generator            : MotionGeneratorForOther    (新規, ランダム初期化)
    value_estimator             : ValueEstimatorMG           (新規, ランダム初期化)
    integration_module          : 飯塚モデルから転送可
    vision_decoder_module       : 飯塚モデルから転送可
    feature_prediction_module   : 飯塚モデルから転送可

野口ら論文との対応:
    motion_generator が スライド 29-33 の MG(LSTM) に相当する。
    MG の隠れ状態（および h²）に「緑を目指す意図」のアトラクタが
    視覚予測学習だけで自己組織化されることを期待する。
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
from modules_mg import MotionGeneratorForOther, SuperpositionModuleWithMG
from value_estimator_mg import ValueEstimatorMG


class SuperpositionNetworkWithMG(nn.Module, WithRNNModuleBase):
    """
    Motion Generator 版 SuperpositionNetworkFeaturePrediction。

    Args:
        config    : 飯塚モデルと共通の設定 (default.yml)
        q_dim     : Q値の次元 (通常 1)
        mg_hidden : MotionGeneratorForOther の LSTM 隠れ次元 (通常 64)
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
        sp_hidden  = config.superposition_module.hidden       # 128
        motion_dim = config.superposition_module.input.motion  # 2

        # Φ_s: Process-2 の om 入力は MG 生成値を受け取る
        self.superposition_module = SuperpositionModuleWithMG(
            config.superposition_module, q_dim=q_dim)
        self.rnn_modules.append(self.superposition_module)

        # MG: h²_{t-1} → om_generated
        self.motion_generator = MotionGeneratorForOther(
            sp_hidden=sp_hidden, mg_hidden=mg_hidden, motion_dim=motion_dim)
        self.rnn_modules.append(self.motion_generator)

        # ValueEstimator: h²_{t-1} → q_other (om 依存なし)
        self.value_estimator = ValueEstimatorMG(state_dim=sp_hidden)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: dict, q_self: torch.Tensor,
                p_mask_vision_self: float, p_mask_vision_other: float):
        """
        Args:
            x['self_vision']  : A-1 の視覚入力  (B, 3, 16, 64)
            x['self_motion']  : A-1 の運動       (B, 2)
            q_self            : A-1 Critic の Q値 (B, 1)
            ※ 'other_motion' は不要。MG が内部生成する。

        Returns:
            pred : {'self_vision': (B, 3, 16, 64)}  次ステップ視覚の予測
            ss   : h¹_t  (B, 128)  Process-1 隠れ状態
            os   : h²_t  (B, 128)  Process-2 隠れ状態
        """
        sv = x['self_vision']
        sm = x['self_motion']

        # ---- VisionEncoder ----
        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)

        # ---- h²_{t-1} を取得（循環参照回避のため detach）----
        oh_prev = self.superposition_module.state['other'].hidden.detach()

        # ---- Motion Generator: h²_{t-1} → om_generated ----
        om_generated = self.motion_generator(oh_prev)

        # ---- ValueEstimator: h²_{t-1} → q_other ----
        q_other = self.value_estimator(oh_prev)

        # ---- マスク（元実装と同様）----
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
        """
        飯塚モデルのチェックポイントから互換モジュールの重みを転送する。
        superposition_module (LSTM 次元変更)、motion_generator、value_estimator
        はランダム初期化のままスキップする。
        """
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
