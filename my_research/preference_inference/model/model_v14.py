"""
v14 models (docs/v14_instructions.md, docs/v14_experiment_log.md). Subclasses of the v12 classes; model_v12.py
is not changed.

SuperpositionNetworkMotionPleasureHoldOtherOneBit (v14-B): the v12 stage-2 model with module G's output reduced
to one scalar u = tanh(.) in (-1, 1); Q2 = u * shape, shape = the fixed v14-A candidate c+ (config
motion_generator_module.shape_vector: bins 0,1 negative, 6,7 positive, 2-5 zero). G is built as in v12
(LSTMCell on the UNMASKED other-encoder output -> fc -> tanh) with output 1. process-2 = [0, 0, Q2, 1].
pred['q2_hat'] is Q2 (8), pred['u'] is u (B, 1).

SuperpositionNetworkMotionPleasureHoldSigned (v14-C): the v12 stage-1 model (SuperpositionNetworkMotionPleasureHold)
with the pleasure sign set per episode: P = s * (Red - Green), s = +1 if A-1 likes Red, -1 if Green
(x['a1_liked_t'], 1 Red / 0 Green, given by exp/loader_v14.py). Without that key s = +1 (= v12).
"""

import torch
import torch.nn.functional as F

from . import util
from .model_v12 import MOTION_DIM, PLEASURE_DIM, SuperpositionNetworkMotionPleasureHold


class SuperpositionNetworkMotionPleasureHoldOtherOneBit(SuperpositionNetworkMotionPleasureHold):
    def __init__(self, config):
        super().__init__(config)
        from .modules import MotionGeneratorModule
        self.motion_generator_module = MotionGeneratorModule(config.motion_generator_module)
        self.rnn_modules.append(self.motion_generator_module)
        assert config.motion_generator_module.output == 1
        shape = torch.tensor([float(v) for v in config.motion_generator_module.shape_vector])
        assert shape.shape == (PLEASURE_DIM,)
        self.register_buffer('q2_shape', shape)

    def forward(self, x, p_mask_vision_self, p_mask_vision_other, p1_override=None, p2_override=None,
                hidden_override=None):
        sv = x['self_vision']
        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)
        p1 = self.process1_input(x, p1_override, hidden_override)
        u = self.motion_generator_module(ov_enc)          # (B, 1); G always runs (keeps its LSTM state)
        q2 = u * self.q2_shape
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
        return {'self_vision': self.vision_decoder_module(so), 'q2_hat': q2, 'u': u}


class SuperpositionNetworkMotionPleasureHoldSigned(SuperpositionNetworkMotionPleasureHold):
    def compute_pleasure(self, sv):
        P = super().compute_pleasure(sv)
        return P if self._sign is None else P * self._sign

    def forward(self, x, *a, **k):
        lk = x.get('a1_liked_t', None)
        self._sign = None if lk is None else (2 * lk.float() - 1).to(x['self_vision'].device)[:, None]
        return super().forward(x, *a, **k)
