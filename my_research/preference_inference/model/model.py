import torch
import torch.nn as nn
import torch.nn.functional as F

from . import util
from .base import WithRNNModuleBase
from .modules import (FeaturePredictionModule, IntegrationModule,
                      MotionGeneratorModule, SuperpositionModule,
                      ValueEstimatorModule, VisionDecoderModule,
                      VisionEncoderModule)


def add_vision_encoder_module(self, config):

    self.share_lns = nn.ModuleList([
        nn.LayerNorm(config.vision_encoder_module.fc[i].output)
        for i in range(len(config.vision_encoder_module.fc))
    ])
    self.self_vision_encoder_module = VisionEncoderModule(
        config.vision_encoder_module, self.share_lns)
    self.other_vision_encoder_module = VisionEncoderModule(
        config.vision_encoder_module, self.share_lns)


def add_integration_module(self, config):
    self.integration_module = IntegrationModule(config.integration_module)


def add_feature_prediction_module(self, config):
    self.feature_prediction_module = FeaturePredictionModule(
        config.feature_prediction_module)


def add_vision_decoder_module(self, config):
    convolved_shape = self.self_vision_encoder_module.get_convolved_shape()
    self.vision_decoder_module = VisionDecoderModule(
        config.vision_decoder_module, convolved_shape)


def add_ae_vision_decoder_module(self, config):
    convolved_shape = self.self_vision_encoder_module.get_convolved_shape()
    self.ae_vision_decoder_module = VisionDecoderModule(
        config.vision_decoder_module, convolved_shape)


def add_superposition_module(self, config):
    self.superposition_module = SuperpositionModule(
        config.superposition_module)
    self.rnn_modules.append(self.superposition_module)


def add_motion_generator_module(self, config):
    self.motion_generator_module = MotionGeneratorModule(
        config.motion_generator_module)
    self.rnn_modules.append(self.motion_generator_module)


def add_value_estimator_module(self, config):
    self.value_estimator_module = ValueEstimatorModule(
        config.value_estimator_module)


class SuperpositionNetworkBase(nn.Module, WithRNNModuleBase):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.rnn_modules = []
        add_vision_encoder_module(self, config)
        add_integration_module(self, config)
        add_vision_decoder_module(self, config)
        add_superposition_module(self, config)


# ---------------------------------------------------------------------------
# Original models (unchanged from base repo)
# ---------------------------------------------------------------------------

class SuperpositionNetwork(SuperpositionNetworkBase):
    def forward(self, x, p_mask_vision_self, p_mask_vision_other):
        sv = x['self_vision']
        sm = x['self_motion']

        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)
        om = torch.zeros_like(sm)

        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        ss, os = self.superposition_module(sv_enc, sm, ov_enc, om)

        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )

        pred = {}
        pred['self_vision'] = self.vision_decoder_module(so)

        return pred


class Autoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        add_vision_encoder_module(self, config)
        add_ae_vision_decoder_module(self, config)

    def forward(self, sv, decode_from_other=False):

        rec = {}

        sv_enc = self.self_vision_encoder_module(sv)
        rec['self_vision'] = self.ae_vision_decoder_module(sv_enc)

        if decode_from_other:
            ov_enc = self.other_vision_encoder_module(sv)
            rec['other_vision'] = self.ae_vision_decoder_module(ov_enc)

        return rec


class SuperpositionNetworkMotionGeneration(SuperpositionNetworkBase):
    def __init__(self, config):
        super().__init__(config)
        add_motion_generator_module(self, config)

    def forward(self, x, p_mask_vision_self, p_mask_vision_other):
        sv = x['self_vision']
        sm = x['self_motion']

        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)

        om = self.motion_generator_module(ov_enc)

        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        ss, os = self.superposition_module(sv_enc, sm, ov_enc, om)

        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )

        pred = {}
        pred['self_vision'] = self.vision_decoder_module(so)
        pred['other_motion'] = om

        return pred


class SuperpositionNetworkFeaturePrediction(SuperpositionNetwork):
    def __init__(self, config):
        super().__init__(config)
        add_feature_prediction_module(self, config)

    def predict_feature(self, x):
        return self.feature_prediction_module(x)


class SuperpositionNetworkMotionGenerationFeaturePrediction(
        SuperpositionNetworkMotionGeneration):
    def __init__(self, config):
        super().__init__(config)
        add_feature_prediction_module(self, config)

    def predict_feature(self, x):
        return self.feature_prediction_module(x)


# ---------------------------------------------------------------------------
# Approach B models (new)
# ---------------------------------------------------------------------------

class SuperpositionNetworkApproachBBase(SuperpositionNetworkBase):
    """
    Approach B base model.
    SM uses 1-dim Q-value input instead of 2-dim action vector.

    Process-1: f¹_v + Q¹_t (1-dim, A-1's true Q-value) → SM
    Process-2: f²_v + 0   (1-dim zero)                  → SM

    Requires model config with superposition_module.input.motion = 1.
    Data must contain 'a1_q_values' field.
    """
    def forward(self, x, p_mask_vision_self, p_mask_vision_other):
        sv = x['self_vision']
        q1 = x['a1_q_values']  # (B, 1)

        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)
        q2 = torch.zeros_like(q1)  # zero for Process-2

        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        ss, os = self.superposition_module(sv_enc, q1, ov_enc, q2)

        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )

        pred = {}
        pred['self_vision'] = self.vision_decoder_module(so)

        return pred


class SuperpositionNetworkApproachBMGVE(SuperpositionNetworkApproachBBase):
    """
    Approach B with Motion Generator (MG) and Value Estimator (VE).

    Process-1: f¹_v + Q¹_t      (1-dim)  → SM
    Process-2: f²_v + Q̂²_t (VE) (1-dim)  → SM

    MG: ov_enc (64-dim) → om_generated (2-dim)
    VE: [ov_enc, om_generated] → Q̂²_t (1-dim)

    During training (always-mask mode), SM Process-2 receives VE's estimate.
    VE is trained indirectly through visual prediction loss only.
    """
    def __init__(self, config):
        super().__init__(config)
        add_motion_generator_module(self, config)
        add_value_estimator_module(self, config)

    def forward(self, x, p_mask_vision_self, p_mask_vision_other):
        sv = x['self_vision']
        q1 = x['a1_q_values']  # (B, 1)

        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)  # unmasked for MG/VE

        om_generated = self.motion_generator_module(ov_enc)  # (B, 2)
        q2_hat = self.value_estimator_module(ov_enc, om_generated)  # (B, 1)

        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        ss, os = self.superposition_module(sv_enc, q1, ov_enc, q2_hat)

        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )

        pred = {}
        pred['self_vision'] = self.vision_decoder_module(so)
        pred['other_motion'] = om_generated
        pred['q2_hat'] = q2_hat

        return pred


# ---------------------------------------------------------------------------
# Approach B v3 models (with FPM)
# ---------------------------------------------------------------------------

class SuperpositionNetworkApproachBBaseV3(SuperpositionNetworkApproachBBase):
    """
    B-base v3: adds FPM to B-base.
    FPM loss: MSE(FPM(ss), sv_enc) + MSE(FPM(os), ov_enc)
    Gradient does NOT flow from FPM back to SM (ss/os detached in runner).
    """
    def __init__(self, config):
        super().__init__(config)
        add_feature_prediction_module(self, config)

    def predict_feature(self, x):
        return self.feature_prediction_module(x)


class SuperpositionNetworkApproachBMGVEV3(SuperpositionNetworkApproachBMGVE):
    """
    B-MGVE v3: adds FPM to B-MGVE.
    FPM loss gradient flows back through os → SM (frozen) → MG/VE input.
    This indirect gradient is the primary learning signal for MG and VE.
    """
    def __init__(self, config):
        super().__init__(config)
        add_feature_prediction_module(self, config)

    def predict_feature(self, x):
        return self.feature_prediction_module(x)
