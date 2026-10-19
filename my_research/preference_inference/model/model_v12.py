"""
v12 stage 1 (docs/v12_experiment_log.md Sec.3-4).

SuperpositionNetworkMotionPleasure = v11's SuperpositionNetworkPleasure with the paper's motion
input restored: process-1 gets [motion (2), pleasure P_t (8), hidden flag (1)] = 11 dims.
  * motion = x['self_motion'] as in the paper (unit step, 0 when asleep), unscaled;
  * P_t = the innate 8-direction pleasure from the *unmasked* current view (as in v11), never hidden;
  * motion is hidden (set to 0, flag = 1) in per-episode intervals: from t = 1 visible and hidden
    intervals alternate (visible first), each of uniform integer length in [len_min, len_max];
    t = 0 is always visible (same construction as v11's option (i),
    model_v11.SuperpositionNetworkPleasureMasked). The schedule is drawn at init_state() from
    torch's CPU generator, so vision masks and dropout (CUDA generator) are unaffected.
process-2 gets 11 zeros (A-2 static, as in the paper's exp1). Prediction target, runner and
feature-prediction monitor are unchanged. No new modules or losses.
Analysis hooks (never used in training): p1_override replaces the 8 pleasure dims,
p2_override replaces the whole 11-dim process-2 input, hidden_override (B,) bool replaces the
schedule at this step, motion_masking (attribute) switches the schedule off.
"""

import torch
import torch.nn.functional as F

from . import util
from .model import SuperpositionNetworkBase, add_feature_prediction_module

from v11core.pleasure import pleasure_torch

MOTION_DIM, PLEASURE_DIM = 2, 8
IN_DIM = MOTION_DIM + PLEASURE_DIM + 1


class SuperpositionNetworkMotionPleasure(SuperpositionNetworkBase):
    def __init__(self, config):
        super().__init__(config)
        add_feature_prediction_module(self, config)
        self.liked = config.pleasure.liked
        self.disliked = config.pleasure.disliked
        self.pleasure_scale = float(config.pleasure.scale)
        assert config.superposition_module.input.motion == IN_DIM
        mm = config.motion_mask
        self.len_min, self.len_max = int(mm.len_min), int(mm.len_max)
        self.motion_masking = bool(mm.enabled)
        self.n_steps = int(mm.n_steps)

    def predict_feature(self, x):
        return self.feature_prediction_module(x)

    def compute_pleasure(self, sv):
        """sv: (B, 3, H, W) scaled to [-1, 1] by the data loader."""
        return self.pleasure_scale * pleasure_torch((sv + 1) / 2, self.liked, self.disliked)

    def make_schedule(self, batch_size):
        hidden = torch.zeros(batch_size, self.n_steps, dtype=torch.bool)
        for b in range(batch_size):
            t, vis = 1, True
            while t < self.n_steps:
                L = int(torch.randint(self.len_min, self.len_max + 1, (1,)).item())
                if not vis:
                    hidden[b, t:t + L] = True
                t += L
                vis = not vis
        return hidden

    def init_state(self, batch_size):
        super().init_state(batch_size)
        self._hidden = self.make_schedule(batch_size)
        self._t = 0

    def current_hidden(self, batch_size, device):
        if self.motion_masking and self._t < self.n_steps:
            return self._hidden[:, self._t].to(device)
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    def process1_input(self, x, p1_override=None, hidden_override=None):
        sv, sm = x['self_vision'], x['self_motion'].float()
        h = self.current_hidden(sv.shape[0], sv.device) if hidden_override is None else hidden_override.to(sv.device)
        P = self.compute_pleasure(sv) if p1_override is None else p1_override
        m = torch.where(h[:, None], torch.zeros_like(sm), sm)
        self.last_hidden = h
        return torch.cat([m, P, h[:, None].float()], 1)

    def forward(self, x, p_mask_vision_self, p_mask_vision_other, p1_override=None, p2_override=None,
                hidden_override=None):
        sv = x['self_vision']
        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)
        p1 = self.process1_input(x, p1_override, hidden_override)
        p2 = torch.zeros_like(p1) if p2_override is None else p2_override
        self._t += 1
        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)
        ss, os = self.superposition_module(sv_enc, p1, ov_enc, p2)
        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )
        return {'self_vision': self.vision_decoder_module(so)}
