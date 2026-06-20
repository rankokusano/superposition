"""
model_phase1.py

Phase 1: Superposition Module (SM) ベーストレーニング。
A-2 静止条件下で SM を学習させる。MG・VE は含まない。

【Phase 2 との重み互換性】
    SuperpositionModuleWithMG (step2/modules_mg.py) を使用。
    Phase 2 ではこの SM 重みを frozen にして MG だけを学習する。

【forward の設計】
    om_generated = zeros(B, 2)   ← A-2 静止なので運動ゼロ
    q_other      = zeros(B, 1)   ← VE なし
    p_mask_self  = 0.0 (step 0), 1.0 (step > 0)  ← 元論文 exp3 準拠
    p_mask_other = 0.0 (step 0), 1.0 (step > 0)  ← 同上
"""

import sys
import os

_HERE     = os.path.dirname(os.path.abspath(__file__))
STEP2_DIR = os.path.abspath(os.path.join(_HERE, '../../step2'))
PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP2_DIR, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import util
from model.base import WithRNNModuleBase
from model.modules import (
    FeaturePredictionModule,
    IntegrationModule,
    VisionDecoderModule,
    VisionEncoderModule,
)
from modules_mg import SuperpositionModuleWithMG


class SuperpositionNetworkPhase1(nn.Module, WithRNNModuleBase):
    """
    Phase 1 用 Superposition Network。

    MG・VE なし。om_generated=0, q_other=0 を固定で SM に渡す。
    SM は SuperpositionModuleWithMG を使用し、Phase 2 と重みを共有できる。

    訓練対象: SM, integration, decoder, feature_prediction
    固定     : both vision encoders (飯塚モデルから転送)
    """

    def __init__(self, config, q_dim: int = 1):
        super().__init__()
        self.config = config
        self.q_dim = q_dim
        self.rnn_modules = []

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

        self.superposition_module = SuperpositionModuleWithMG(
            config.superposition_module, q_dim=q_dim)
        self.rnn_modules.append(self.superposition_module)

    def forward(self, x: dict, q_self: torch.Tensor,
                p_mask_vision_self: float, p_mask_vision_other: float):
        """
        x['self_vision'] : (B, 3, 16, 64)
        x['self_motion'] : (B, 2)
        q_self           : (B, 1)
        return: pred dict, h¹ (B, 128), h² (B, 128)
        """
        sv = x['self_vision']
        sm = x['self_motion']
        B  = sv.shape[0]
        device = sv.device

        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)

        om_zero     = torch.zeros(B, 2, device=device)
        q_other_zero = torch.zeros(B, self.q_dim, device=device)

        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        ss, os_ = self.superposition_module(
            sv_enc, sm, q_self,
            ov_enc, om_zero, q_other_zero,
        )

        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os_, p=0.5, training=self.training),
        )
        pred = {'self_vision': self.vision_decoder_module(so)}

        return pred, ss, os_

    def predict_feature(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_prediction_module(x)

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
