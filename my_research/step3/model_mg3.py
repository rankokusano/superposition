"""
model_mg3.py

step2/model_mg2.py (SuperpositionNetworkWithMG2) からの変更点:

    【変更箇所】ValueEstimator の入力

        変更前 (mg2): q_other = value_estimator(h²_{t-1})
                      → V(h²) 型: 過去状態から推測
                      → 循環依存あり

        変更後 (mg3): q_other = value_estimator(ov_enc, om_generated)
                      → Q(s,a) 型: 現ステップの特徴量から推測
                      → 循環依存なし

    【変更しないもの】
        - SuperpositionModuleWithMG  (step2/modules_mg.py をそのまま利用)
        - MotionGeneratorForOtherV2  (step2/modules_mg2.py をそのまま利用)
        - 飯塚互換モジュール群        (VisionEncoder, Integration, Decoder 等)
        - load_iizuka_weights()      (重み転送ロジックは同じ)

    【forward の変更点】
        変更前:
            oh_prev = self.superposition_module.state['other'].hidden.detach()
            q_other = self.value_estimator(oh_prev)

        変更後:
            q_other = self.value_estimator(ov_enc, om_generated)
            ← MG と同じタイミング（マスク前の ov_enc）を使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

_HERE     = os.path.dirname(os.path.abspath(__file__))
STEP2_DIR = os.path.abspath(os.path.join(_HERE, '../step2'))
PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '../..'))
for p in [_HERE, STEP2_DIR, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import util
from model.base import WithRNNModuleBase
from model.modules import (
    FeaturePredictionModule,
    IntegrationModule,
    VisionDecoderModule,
    VisionEncoderModule,
)
from modules_mg import SuperpositionModuleWithMG
from modules_mg2 import MotionGeneratorForOtherV2
from value_estimator_v2 import ValueEstimatorV2


class SuperpositionNetworkWithMG3(nn.Module, WithRNNModuleBase):
    """
    ValueEstimator 入力を (ov_enc, om_generated) に変更した SuperpositionNetwork。

    step2/model_mg2.py (SuperpositionNetworkWithMG2) との差分:
        value_estimator クラス : ValueEstimatorMG  → ValueEstimatorV2
        value_estimator 入力   : h²_{t-1} (128-dim) → (ov_enc, om_generated) (64+2-dim)
    """

    def __init__(self, config, q_dim: int = 1, mg_hidden: int = 64):
        super().__init__()
        self.config = config
        self.q_dim = q_dim
        self.rnn_modules = []

        # ---- 飯塚互換モジュール（step2/model_mg2.py と同じ） ----
        self.share_lns = nn.ModuleList([
            nn.LayerNorm(config.vision_encoder_module.fc[i].output)
            for i in range(len(config.vision_encoder_module.fc))
        ])
        self.self_vision_encoder_module = VisionEncoderModule(
            config.vision_encoder_module, self.share_lns)
        self.other_vision_encoder_module = VisionEncoderModule(
            config.vision_encoder_module, self.share_lns)
        self.integration_module    = IntegrationModule(config.integration_module)
        self.vision_decoder_module = VisionDecoderModule(
            config.vision_decoder_module,
            self.self_vision_encoder_module.get_convolved_shape())
        self.feature_prediction_module = FeaturePredictionModule(
            config.feature_prediction_module)

        # ---- 新規モジュール ----
        enc_dim    = config.superposition_module.input.vision  # 64
        motion_dim = config.superposition_module.input.motion  # 2

        self.superposition_module = SuperpositionModuleWithMG(
            config.superposition_module, q_dim=q_dim)
        self.rnn_modules.append(self.superposition_module)

        self.motion_generator = MotionGeneratorForOtherV2(
            enc_dim=enc_dim, mg_hidden=mg_hidden, motion_dim=motion_dim)
        self.rnn_modules.append(self.motion_generator)

        # ValueEstimator: (ov_enc, om_generated) → q_other  ← mg2 からの変更点
        self.value_estimator = ValueEstimatorV2(
            enc_dim=enc_dim, motion_dim=motion_dim)

    def forward(self, x: dict, q_self: torch.Tensor,
                p_mask_vision_self: float, p_mask_vision_other: float):
        """
        x['self_vision']  : (B, 3, 16, 64)
        x['self_motion']  : (B, 2)
        q_self            : (B, 1)

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

        # ---- Motion Generator: ov_enc → om_generated ----
        om_generated = self.motion_generator(ov_enc)

        # ---- ValueEstimator: (ov_enc, om_generated) → q_other ----
        # マスク前の ov_enc を使う（MG と同じタイミング）
        # 循環依存なし: ov_enc は encoder の出力であり h² に依存しない
        q_other = self.value_estimator(ov_enc, om_generated)

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

    def load_iizuka_weights(self, checkpoint: dict, device):
        """step2/model_mg2.py と同じロジック。変更なし。"""
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
