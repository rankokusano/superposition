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


class SuperpositionNetworkMotionPleasureHold(SuperpositionNetworkMotionPleasure):
    """
    v12 trial (P-c) and later (docs/v12_experiment_log.md Sec.10): as SuperpositionNetworkMotionPleasure,
    but while the motion is hidden the pleasure input is held at its value at the FIRST hidden step of
    the interval (computed from that step's view, which precedes the hidden motion), so that changes
    of P cannot reveal the hidden motion. Visible steps use the current P. p1_override (analysis)
    bypasses the hold.
    """

    def init_state(self, batch_size):
        super().init_state(batch_size)
        self._held = None
        self._prev_hidden = None

    def process1_input(self, x, p1_override=None, hidden_override=None):
        sv, sm = x['self_vision'], x['self_motion'].float()
        h = self.current_hidden(sv.shape[0], sv.device) if hidden_override is None else hidden_override.to(sv.device)
        if p1_override is not None:
            P = p1_override
        else:
            P_now = self.compute_pleasure(sv)
            prev = torch.zeros_like(h) if self._prev_hidden is None else self._prev_hidden
            start = h & ~prev
            held = P_now if self._held is None else torch.where(start[:, None], P_now, self._held)
            P = torch.where(h[:, None], held, P_now)
            self._held = held
        self._prev_hidden = h
        m = torch.where(h[:, None], torch.zeros_like(sm), sm)
        self.last_hidden = h
        return torch.cat([m, P, h[:, None].float()], 1)


class SuperpositionNetworkMotionPleasureHoldOther(SuperpositionNetworkMotionPleasureHold):
    """
    v12 stage 2 (docs/v12_experiment_log.md Sec.11.3, 21): the stage-1 model plus the other's module G,
    built like the paper's MotionGeneratorModule (LSTMCell on the UNMASKED other-encoder output -> fc
    -> tanh), 8-dim output Q2. process-2 = [motion 0, pleasure = Q2, flag 1]. process-1's motion is not
    hidden in stage 2 (config motion_mask.enabled = false; flag 0, current pleasure). Trained with the
    paper's exp3 recipe (only G trainable). p2_override (analysis) replaces the whole 11-dim process-2
    input; the returned pred['q2_hat'] is always G's output.
    """

    def __init__(self, config):
        super().__init__(config)
        from .modules import MotionGeneratorModule
        self.motion_generator_module = MotionGeneratorModule(config.motion_generator_module)
        self.rnn_modules.append(self.motion_generator_module)
        assert config.motion_generator_module.output == PLEASURE_DIM

    def forward(self, x, p_mask_vision_self, p_mask_vision_other, p1_override=None, p2_override=None,
                hidden_override=None):
        sv = x['self_vision']
        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)
        p1 = self.process1_input(x, p1_override, hidden_override)
        q2 = self.motion_generator_module(ov_enc)          # G always runs (keeps its LSTM state)
        if p2_override is None:
            p2 = torch.cat([torch.zeros_like(q2[:, :MOTION_DIM]), q2, torch.ones_like(q2[:, :1])], 1)
        else:
            p2 = p2_override
        self._t += 1
        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)
        ss, os = self.superposition_module(sv_enc, p1, ov_enc, p2)
        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )
        return {'self_vision': self.vision_decoder_module(so), 'q2_hat': q2}


class SuperpositionNetworkMotionPleasureHoldOtherShaped(SuperpositionNetworkMotionPleasureHoldOther):
    """
    v12 stage 2 with G's output restricted to the form of the pleasure the network itself experienced in
    stage 1 (docs/v12_experiment_log.md Sec.34; a design changed after seeing the stage-2 results). No new
    module or loss: only G's 8 tanh outputs z are mapped, by config motion_generator_module.shape:
      'mask'         : Q2 = P_max * z on bins 0,1,6,7 and 0 on bins 2-5 (in stage 1 the pleasure is always 0
                       on bins 2-5 and |P| <= P_max = 1 on the others);
      'signed_pairs' : the set of pleasure vectors seen in stage 1: bins 2-5 = 0, bins {0,1} share one sign and
                       bins {6,7} the opposite sign. With u = z[2] (sign / strength, in (-1, 1)) and
                       m_k = (z_k + 1) / 2 in (0, 1) for k in {0,1,6,7}:
                       Q2 = P_max * [-u m0, -u m1, 0, 0, 0, 0, u m6, u m7].
    pred['q2_hat'] is the restricted Q2 (what enters process-2).
    """
    COLOUR_BINS = (0, 1, 6, 7)

    def __init__(self, config):
        super().__init__(config)
        self.q2_shape = str(config.motion_generator_module.shape)
        self.p_max = float(config.motion_generator_module.p_max)
        assert self.q2_shape in ('mask', 'signed_pairs')
        mask = torch.zeros(PLEASURE_DIM)
        mask[list(self.COLOUR_BINS)] = 1.0
        self.register_buffer('q2_mask', mask)

    def shape_q2(self, z):
        if self.q2_shape == 'mask':
            return self.p_max * z * self.q2_mask
        u = z[:, 2:3]
        m = (z + 1) / 2
        q = torch.zeros_like(z)
        q[:, 0:2] = -u * m[:, 0:2]
        q[:, 6:8] = u * m[:, 6:8]
        return self.p_max * q

    def forward(self, x, p_mask_vision_self, p_mask_vision_other, p1_override=None, p2_override=None,
                hidden_override=None):
        sv = x['self_vision']
        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)
        p1 = self.process1_input(x, p1_override, hidden_override)
        q2 = self.shape_q2(self.motion_generator_module(ov_enc))
        if p2_override is None:
            p2 = torch.cat([torch.zeros_like(q2[:, :MOTION_DIM]), q2, torch.ones_like(q2[:, :1])], 1)
        else:
            p2 = p2_override
        self._t += 1
        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)
        ss, os = self.superposition_module(sv_enc, p1, ov_enc, p2)
        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )
        return {'self_vision': self.vision_decoder_module(so), 'q2_hat': q2}
