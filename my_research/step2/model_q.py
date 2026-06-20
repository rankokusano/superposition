"""
model_q.py

飯塚ネットワーク (SuperpositionNetworkFeaturePrediction) のQ値入力拡張版。

変更点のまとめ:
    1. SuperpositionModule → SuperpositionModuleWithQ (LSTM入力+1次元)
    2. ValueEstimator をメンバーとして追加
    3. forward() が q_self (A-1 Criticからの真のQ値) を受け取る
    4. q_other (A-2 推測Q値) は ValueEstimator が内部で計算
    5. other_motion を入力辞書に追加 (x['other_motion'])

再学習に関わる部分はフラグで切り替えられるよう、
パラメータを get_trainable_params() で分けて返す。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, "/work")
from model import util
from model.base import WithRNNModuleBase
from model.modules import (
    FeaturePredictionModule,
    IntegrationModule,
    VisionDecoderModule,
    VisionEncoderModule,
)
from modules_q import SuperpositionModuleWithQ
from value_estimator import ValueEstimator


class SuperpositionNetworkWithQ(nn.Module, WithRNNModuleBase):
    """
    Q値入力付きSuperpositionNetworkFeaturePrediction。

    モジュール構成:
        self_vision_encoder_module  : 飯塚ネットワークと共通 (転送可)
        other_vision_encoder_module : 飯塚ネットワークと共通 (転送可)
        share_lns                   : 飯塚ネットワークと共通 (転送可)
        superposition_module        : Q入力追加のため LSTM次元変更 → ランダム初期化
        integration_module          : 飯塚ネットワークと共通 (転送可)
        vision_decoder_module       : 飯塚ネットワークと共通 (転送可)
        feature_prediction_module   : 飯塚ネットワークと共通 (転送可)
        value_estimator             : 新規モジュール (ランダム初期化)
    """

    def __init__(self, config, q_dim: int = 1):
        super().__init__()
        self.config = config
        self.q_dim = q_dim

        self.rnn_modules = []

        # ---------- 飯塚ネットワークと同構造のモジュール群 ----------
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

        # ---------- Q値入力対応 SuperpositionModule ----------
        self.superposition_module = SuperpositionModuleWithQ(
            config.superposition_module, q_dim=q_dim)
        self.rnn_modules.append(self.superposition_module)

        # ---------- 新規: A-2 の Q値推測モジュール ----------
        sp_hidden = config.superposition_module.hidden  # 128
        motion_dim = config.superposition_module.input.motion  # 2
        self.value_estimator = ValueEstimator(
            state_dim=sp_hidden, action_dim=motion_dim)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: dict, q_self: torch.Tensor,
                p_mask_vision_self: float, p_mask_vision_other: float):
        """
        x['self_vision']  : A-1 の視覚入力  (B, 3, 16, 64)
        x['self_motion']  : A-1 の運動       (B, 2)
        x['other_motion'] : A-2 の運動       (B, 2)
        q_self            : A-1 Critic の出力Q値  (B, q_dim=1)

        return:
            pred : {'self_vision': (B, 3, 16, 64)}  — 次ステップ視覚の予測
            ss   : h¹_t  (B, 128)  — Process-1 隠れ状態
            os   : h²_t  (B, 128)  — Process-2 隠れ状態
        """
        sv = x['self_vision']    # A-1 の視覚
        sm = x['self_motion']    # A-1 の運動
        om = x['other_motion']   # A-2 の運動 (GreenFollower)

        # --- VisionEncoder ---
        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)

        # --- Process-2 の Q値推測 ---
        # h²_{t-1} (前ステップ隠れ状態) を使い循環参照を回避
        oh_prev = self.superposition_module.state['other'].hidden.detach()
        q_other = self.value_estimator(oh_prev, om)  # (B, 1)

        # --- マスク (元実装と同様) ---
        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        # --- Φ_s ---
        ss, os = self.superposition_module(
            sv_enc, sm, q_self,
            ov_enc, om, q_other,
        )

        # --- Integration → VisionDecoder ---
        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )
        pred = {'self_vision': self.vision_decoder_module(so)}

        return pred, ss, os

    def predict_feature(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_prediction_module(x)

    # ------------------------------------------------------------------
    # 学習対象パラメータの分割
    # ------------------------------------------------------------------

    def get_trainable_params(self, retrain_iizuka: bool = False):
        """
        retrain_iizuka=False : Φ_s (SuperpositionModuleWithQ) と
                               ValueEstimator のパラメータのみ返す
        retrain_iizuka=True  : 全パラメータを返す
        """
        if retrain_iizuka:
            return list(self.parameters())

        core_params = (
            list(self.superposition_module.parameters()) +
            list(self.value_estimator.parameters())
        )
        return core_params

    # ------------------------------------------------------------------
    # 事前学習済み重みのロード (LSTM は互換性なしのため strict=False)
    # ------------------------------------------------------------------

    def load_iizuka_weights(self, checkpoint: dict, device):
        """
        飯塚モデルのチェックポイントから互換モジュールの重みだけを転送する。
        SuperpositionModule の LSTM は入力次元が変わっているためスキップ。
        checkpoint は torch.load() の戻り値 (トップレベルに 'model' キーを持つ)。
        """
        # チェックポイント形式の正規化
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
        loaded = 0
        skipped = 0
        for k, v in state_dict.items():
            if any(k.startswith(prefix) for prefix in compatible_keys):
                if k in own_dict and own_dict[k].shape == v.shape:
                    own_dict[k] = v
                    loaded += 1
                else:
                    skipped += 1
            else:
                skipped += 1
        self.load_state_dict(own_dict)
        print(f"[load_iizuka_weights] loaded={loaded}, skipped={skipped}")
