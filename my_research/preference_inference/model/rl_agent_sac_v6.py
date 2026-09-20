"""
v6 reward-conditioned SAC actor/critic: Q(s, a, r) and pi(a | s, r).

This is a copy-then-extend of my_research/rl_agent_sac.py, not an in-place
edit of it. rl_agent_sac.py is imported by 70+ files across the repo,
including my_research/step3/ (a separate, protected research line) -- see
docs/v6_experiment_log.md §3 for why editing it directly was rejected.

Design (docs/v6_instructions.md §2.2-2.3, decided in
docs/v6_experiment_log.md §3):
  - r = (w_red, w_green, w_blue, w_cyan), a 4-dim reward-parameter vector.
    A-1's true r = (+1, -1, 0, 0); A-2's true r = (-1, +1, 0, 0).
  - Both the critic Q(s, a, r) and the actor pi(a | s, r) are conditioned
    on r (the actor needs it too, or the SAC target `r_reward + gamma *
    (Q'(s', a') - alpha * logpi(a'|s'))` bootstraps through an
    r-independent action, biasing the learned Q).
  - r is concatenated at the same point as the action (after the LSTM),
    not fed into the LSTM input alongside the CNN features: r is constant
    for a whole episode (no temporal structure to integrate), unlike the
    visual observations the LSTM is there to integrate.
  - film=True (2026-09-21, docs/v6_experiment_log.md §17.3) additionally
    modulates the 256-d CNN features by r before the LSTM (FiLM). Off by
    default, so existing checkpoints and runs are unaffected.
"""
import numpy as np
import torch
import torch.nn as nn

LOG_STD_MAX = 2
LOG_STD_MIN = -20

R_DIM = 4
LANDMARK_ORDER = ('red', 'green', 'blue', 'cyan')
A1_TRUE_R = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
A2_TRUE_R = np.array([-1.0, 1.0, 0.0, 0.0], dtype=np.float32)


class CNNEncoder(nn.Module):
    """画像から特徴量を抽出するCNN（rl_agent_sac.pyと同一）"""
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 16, 256),
            nn.ReLU(),
        )

    def forward(self, v):
        x = self.cnn(v)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FiLM(nn.Module):
    """
    Feature-wise linear modulation by the reward parameter r
    (docs/v6_experiment_log.md §17.3): out = (1 + W_g r) * feat + W_b r.
    Linear in r, so the interaction feat_j * (sum_k u_jk r_k) is bilinear
    in (feature, reward weights) -- the structure Q ~ sum_k w_k G_k(s, a).
    Zero-initialized, so at the start of training it is the identity and
    the network is exactly the non-FiLM one.
    """
    def __init__(self, r_dim, feat_dim):
        super().__init__()
        self.gamma = nn.Linear(r_dim, feat_dim)
        self.beta = nn.Linear(r_dim, feat_dim)
        for m in (self.gamma, self.beta):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, feat, r):
        return (1.0 + self.gamma(r)) * feat + self.beta(r)


class ActorLSTM(nn.Module):
    """
    行動を決めるネットワーク（Actor）: pi(a | s, r)
    CNN + LSTM -> [lstm_out, r] を連結 -> 平均・標準偏差を出力 -> 連続行動をサンプリング
    """
    def __init__(self, hidden_dim=128, r_dim=R_DIM, film=False):
        super().__init__()
        self.encoder = CNNEncoder()
        self.film = FiLM(r_dim, 256) if film else None
        self.lstm = nn.LSTM(256, hidden_dim, batch_first=True)
        self.mean = nn.Linear(hidden_dim + r_dim, 2)
        self.log_std = nn.Linear(hidden_dim + r_dim, 2)
        self.hidden_dim = hidden_dim
        self.r_dim = r_dim

    def forward(self, v, r, hidden=None):
        feat = self.encoder(v)
        if self.film is not None:
            feat = self.film(feat, r)
        feat = feat.unsqueeze(1)
        lstm_out, hidden = self.lstm(feat, hidden)
        lstm_out = lstm_out.squeeze(1)
        x = torch.cat([lstm_out, r], dim=-1)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, hidden

    def sample(self, v, r, hidden=None):
        mean, log_std, hidden = self.forward(v, r, hidden)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, hidden


class CriticLSTM(nn.Module):
    """
    Q値を評価するネットワーク（Critic）: Q(s, a, r)
    CNN + LSTM + 行動 + 報酬パラメータ -> Q値
    """
    def __init__(self, hidden_dim=128, r_dim=R_DIM, film=False):
        super().__init__()
        self.encoder = CNNEncoder()
        self.film = FiLM(r_dim, 256) if film else None
        self.lstm = nn.LSTM(256, hidden_dim, batch_first=True)
        self.q = nn.Sequential(
            nn.Linear(hidden_dim + 2 + r_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.r_dim = r_dim

    def forward(self, v, action, r, hidden=None):
        feat = self.encoder(v)
        if self.film is not None:
            feat = self.film(feat, r)
        feat = feat.unsqueeze(1)
        lstm_out, hidden = self.lstm(feat, hidden)
        lstm_out = lstm_out.squeeze(1)
        x = torch.cat([lstm_out, action, r], dim=-1)
        q = self.q(x)
        return q, hidden


def sample_reward_param(rng=None, eps=1e-6):
    """
    Continuous-uniform sample of a 4-dim reward-parameter vector, then
    L1-normalized to sum(|w_k|) == 2 (docs/v6_experiment_log.md §3,
    2026-09-13 decision). L1 (not L2) was chosen because A-1's true r
    (+1,-1,0,0) and A-2's true r (-1,+1,0,0) already satisfy
    sum(|w_k|)==2, so they are fixed points of this normalization and no
    downstream code needs to track a rescaled "true" value.

    Note: sampling uniformly in the cube then radially projecting onto the
    L1=2 shell is NOT a uniform distribution over that shell (some bias
    toward vertices) -- documented simplification, not claimed rigorous.
    """
    rng = rng if rng is not None else np.random
    w = rng.uniform(-1.0, 1.0, size=R_DIM).astype(np.float32)
    l1 = np.abs(w).sum()
    while l1 < eps:
        w = rng.uniform(-1.0, 1.0, size=R_DIM).astype(np.float32)
        l1 = np.abs(w).sum()
    return w * (2.0 / l1)


def landmark_fractions(vision_np):
    """
    (red, green, blue, cyan) pixel fractions of an HxWx3 float [0,1] vision
    frame. Thresholds (>0.9 for the lit channel(s), <0.1 for the others)
    match the existing red/green convention in
    simulation/train_rl_v3.py:get_reward, and were verified empirically
    (not assumed) against data/data/grid/data.h5 (self_vision_no_agent):
    the four landmark materials render as clean, unblended pixel clusters
    at exactly (1,0,0) red, (0,1,0) green, (0,0,1) blue, (0,1,1) cyan
    (the landmark named "Yellow" in code is confirmed cyan at the pixel
    level, consistent with docs/v4_experiment_log.md's earlier finding),
    fading toward each material's base Kd color (e.g. (0.8,0,0) for red)
    at the edges -- i.e. the same "empty region -> full corner" spatial
    gradient the reward function is built around already exists per-channel
    for all four colors, not just red/green.
    """
    r, g, b = vision_np[:, :, 0], vision_np[:, :, 1], vision_np[:, :, 2]
    total = vision_np.shape[0] * vision_np.shape[1]
    red = ((r > 0.9) & (g < 0.1) & (b < 0.1)).sum() / total
    green = ((g > 0.9) & (r < 0.1) & (b < 0.1)).sum() / total
    blue = ((b > 0.9) & (r < 0.1) & (g < 0.1)).sum() / total
    cyan = ((g > 0.9) & (b > 0.9) & (r < 0.1)).sum() / total
    return np.array([red, green, blue, cyan], dtype=np.float32)


def reward_from_vision(vision_np, w):
    """reward = sum_k w_k * fraction_k. Generalizes get_reward() in
    train_rl_v3.py (reward = red_fraction - green_fraction), which is the
    special case w=(+1,-1,0,0) -- confirmed in
    docs/v6_experiment_log.md §4.1."""
    return float(np.dot(w, landmark_fractions(vision_np)))
