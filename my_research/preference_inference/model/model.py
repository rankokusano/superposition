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


# ---------------------------------------------------------------------------
# R3: probe-Q-vector replaces m_t (2026-08-14)
# ---------------------------------------------------------------------------

class SuperpositionNetworkProbeQ(SuperpositionNetworkBase):
    """
    R3-direct (R3-A): process-1's SM input is a K-dim normalized probe-Q
    vector -- Q(self_vision, a_probe) for K fixed directional actions,
    evaluated by A-1's own frozen, pretrained critic and normalized with
    A-1's own real-data (mu, sigma) via tanh -- computed fresh every
    forward pass, never baked into the dataset (config.probe_q.*).
    process-2 gets a zero vector of the same dimension (post-normalization
    "no directional preference", mirroring how the original paper's base
    training left process-2 at a constant zero motion). No MG, no VE.

    SM's LSTM input width changes from m_t's 2-dim to K-dim, so its
    weights are NOT shape-compatible with an exp1_l1 pretrain checkpoint
    -- util.load_pretrain() skips shape-mismatched keys, so SM trains
    from scratch while vision_encoder/decoder/integration/FPM transfer
    from exp1_l1 unchanged (see exp config's freeze list: everything
    except superposition_module is frozen).
    """
    def __init__(self, config):
        super().__init__(config)
        add_feature_prediction_module(self, config)

        import sys
        if '/work' not in sys.path:
            sys.path.insert(0, '/work')
        from my_research.rl_agent_sac import CriticLSTM

        self.probe_critic = CriticLSTM()
        self.probe_critic.load_state_dict(
            torch.load(config.probe_q.critic_path, map_location='cpu'))
        self.probe_critic.eval()
        for p in self.probe_critic.parameters():
            p.requires_grad = False

        import math
        k = config.probe_q.k
        angles = [2 * math.pi * i / k for i in range(k)]
        probes = torch.tensor(
            [[math.cos(a), math.sin(a)] for a in angles],
            dtype=torch.float32)
        self.register_buffer('probe_actions', probes)
        self.register_buffer('q_mu', torch.tensor(float(config.probe_q.mu)))
        self.register_buffer('q_sigma', torch.tensor(float(config.probe_q.sigma)))

    def predict_feature(self, x):
        return self.feature_prediction_module(x)

    def compute_probe_q(self, sv_raw):
        """sv_raw: (B, 3, H, W) in [0,1] (critic's native scale)."""
        b = sv_raw.size(0)
        k = self.probe_actions.size(0)
        v_rep = sv_raw.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(
            b * k, *sv_raw.shape[1:])
        a_rep = self.probe_actions.unsqueeze(0).expand(b, -1, -1).reshape(b * k, 2)
        with torch.no_grad():
            q, _ = self.probe_critic(v_rep, a_rep, hidden=None)
        q = q.reshape(b, k)
        return torch.tanh((q - self.q_mu) / self.q_sigma)

    def forward(self, x, p_mask_vision_self, p_mask_vision_other):
        sv = x['self_vision']  # already scaled to [-1, 1] by the data loader

        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)

        sv_raw = (sv + 1) / 2  # back to the critic's native [0,1] scale
        q1_vec = self.compute_probe_q(sv_raw)
        q2_vec = torch.zeros_like(q1_vec)

        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        ss, os = self.superposition_module(sv_enc, q1_vec, ov_enc, q2_vec)

        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )

        pred = {}
        pred['self_vision'] = self.vision_decoder_module(so)

        return pred


class SuperpositionNetworkProbeQConcat(SuperpositionNetworkBase):
    """
    R3-curriculum stages b/c: process-1 input is concat([action(vx,vy),
    probe-Q(K)]) -- action_dim(2) + K. The action component is
    stochastically zeroed at rate config.probe_q.action_dropout_rate
    ("time to time no action, only Q" per the professor's note):
    rate=0.0 -> R3b (pure concat, action always present), rate>0 -> R3c
    (increasingly Q-reliant). Applied unconditionally (train and eval),
    since this is a literal input-corruption curriculum condition, not a
    regularizer. process-2 stays a zero vector of the same combined
    dimension throughout (no MG/VE anywhere in R3). No motion generation:
    this class never predicts other_motion.
    """
    def __init__(self, config):
        super().__init__(config)
        add_feature_prediction_module(self, config)

        import sys
        if '/work' not in sys.path:
            sys.path.insert(0, '/work')
        from my_research.rl_agent_sac import CriticLSTM

        self.probe_critic = CriticLSTM()
        self.probe_critic.load_state_dict(
            torch.load(config.probe_q.critic_path, map_location='cpu'))
        self.probe_critic.eval()
        for p in self.probe_critic.parameters():
            p.requires_grad = False

        import math
        k = config.probe_q.k
        angles = [2 * math.pi * i / k for i in range(k)]
        probes = torch.tensor(
            [[math.cos(a), math.sin(a)] for a in angles],
            dtype=torch.float32)
        self.register_buffer('probe_actions', probes)
        self.register_buffer('q_mu', torch.tensor(float(config.probe_q.mu)))
        self.register_buffer('q_sigma', torch.tensor(float(config.probe_q.sigma)))
        self.action_dropout_rate = float(config.probe_q.action_dropout_rate)

    def predict_feature(self, x):
        return self.feature_prediction_module(x)

    def compute_probe_q(self, sv_raw):
        b = sv_raw.size(0)
        k = self.probe_actions.size(0)
        v_rep = sv_raw.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(
            b * k, *sv_raw.shape[1:])
        a_rep = self.probe_actions.unsqueeze(0).expand(b, -1, -1).reshape(b * k, 2)
        with torch.no_grad():
            q, _ = self.probe_critic(v_rep, a_rep, hidden=None)
        q = q.reshape(b, k)
        return torch.tanh((q - self.q_mu) / self.q_sigma)

    def forward(self, x, p_mask_vision_self, p_mask_vision_other):
        sv = x['self_vision']
        sm = x['self_motion']  # (B, 2), raw action

        sv_enc = self.self_vision_encoder_module(sv)
        ov_enc = self.other_vision_encoder_module(sv)

        sv_raw = (sv + 1) / 2
        q1_vec = self.compute_probe_q(sv_raw)

        if self.action_dropout_rate > 0:
            drop = (torch.rand(sm.size(0), 1, device=sm.device)
                    < self.action_dropout_rate).float()
            sm = sm * (1 - drop)

        process1_input = torch.cat([sm, q1_vec], dim=1)
        process2_input = torch.zeros_like(process1_input)

        sv_enc = util.mask(sv_enc, p_mask_vision_self)
        ov_enc = util.mask(ov_enc, p_mask_vision_other)

        ss, os = self.superposition_module(sv_enc, process1_input, ov_enc, process2_input)

        so = self.integration_module(
            F.dropout(ss, p=0.5, training=self.training),
            F.dropout(os, p=0.5, training=self.training),
        )

        pred = {}
        pred['self_vision'] = self.vision_decoder_module(so)

        return pred
