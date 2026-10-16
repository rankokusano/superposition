"""
v11 stage 1 (docs/v11_experiment_log.md Sec.4.4, 4.10, 4.11).

SuperpositionNetworkPleasure = the paper's SuperpositionNetworkFeaturePrediction
with process-1's motion input replaced by the innate 8-direction pleasure
signal P_t computed in forward() from the *unmasked* current view v_t
(v11core/pleasure.py; liked/disliked colours from the model config), and
process-2 fed a zero vector of the same width (A-2 static, as in exp1).
No motion enters the network; the prediction target is the next view only
(+ the paper's detached feature-prediction monitor via the unchanged
PredictionFeaturePredictionRunner). No critic. Trained from scratch.
"""

import torch
import torch.nn.functional as F

from . import util
from .model import SuperpositionNetworkBase, add_feature_prediction_module

from v11core.pleasure import pleasure_torch


class SuperpositionNetworkPleasure(SuperpositionNetworkBase):
    def __init__(self, config):
        super().__init__(config)
        add_feature_prediction_module(self, config)
        self.liked = config.pleasure.liked
        self.disliked = config.pleasure.disliked
        self.pleasure_scale = float(config.pleasure.scale)
        assert config.superposition_module.input.motion == 8

    def predict_feature(self, x):
        return self.feature_prediction_module(x)

    def compute_pleasure(self, sv):
        """sv: (B, 3, H, W) scaled to [-1, 1] by the data loader."""
        sv_raw = (sv + 1) / 2
        return self.pleasure_scale * pleasure_torch(sv_raw, self.liked, self.disliked)

    def forward(self, x, p_mask_vision_self, p_mask_vision_other, p1_override=None):
        sv = x['self_vision']

        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)

        # p1_override: analysis-only hook (M1(b) input manipulation); never
        # used in training.
        p1 = self.compute_pleasure(sv) if p1_override is None else p1_override
        p2 = torch.zeros_like(p1)

        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        ss, os = self.superposition_module(sv_enc, p1, ov_enc, p2)

        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )

        pred = {}
        pred['self_vision'] = self.vision_decoder_module(so)
        return pred
