"""
Shared pieces of the v7 Bayesian inverse planning (docs/v7_experiment_log.md
§2.2, §2.5-§2.7), used by stage 1 (true A-2 vision) and stage 2 (viewpoint
extracted from A-1's vision).

    pi(a | s, r, beta) ∝ exp(beta * Q(s, a, r))      over 64 unit directions
    log pi(a_obs)      = beta*Q(s, â_obs, r) - logsumexp_k beta*Q(s, a_k, r)
    P(r | 1:t)         ∝ P(r) * sum_beta P(beta) * prod_{k<t} pi(a_k | s_k, r, beta)

Q comes from A-1's frozen reward-conditioned critic (v6 seed2), called one
frame at a time (hidden=None), as in the v4-v6 probe-Q. r² (the true goal)
never enters any function here except `evaluate`, which scores predictions.
"""
import math
import os
import sys

import numpy as np
import torch
from scipy.special import logsumexp

sys.path.insert(0, '/work')
from my_research.preference_inference.model.rl_agent_sac_v6 import (  # noqa
    CriticLSTM, landmark_fractions,
)

PI_ROOT = '/work/my_research/preference_inference'
CRITIC_PATH = os.path.join(PI_ROOT, 'data', 'model', 'v6_rl_critic_seed2_film_norelabel_curriculum.pth')

# Candidate order: the 3 data goals first (indices match goal_index), Blue last
# (secondary 4-candidate analysis only, §2.6).
CAND_NAMES = ('Red', 'Green', 'Cyan', 'Blue')
CAND_R = np.array([
    [1.0, -1.0, 0.0, 0.0],
    [-1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 1.0],
    [0.0, 0.0, 1.0, -1.0],
], dtype=np.float32)
N_MAIN = 3
R1_INDEX = 0  # A-1's own r1 = (+1,-1,0,0) = Red goal

BETA_MAIN = (0.5, 1, 2, 4, 8, 16)
BETA_EXT = (0.5, 1, 2, 4, 8, 16, 32, 64, 128)
K_DIRS = 64
SMALL_ACTION = 0.1
T_EVAL = (1, 5, 10, 20, 50, 100)
NEAR_DIST = 5.0
DIST_BINS = ((0, 5), (5, 10), (10, 15), (15, np.inf))
BIN_NAMES = ('<5', '5-10', '10-15', '>=15')
GOAL_POS = np.array([[-9.0, 9.0], [-9.0, -9.0], [9.0, 9.0]], dtype=np.float32)

PROBE_DIRS = np.array([[math.cos(2 * math.pi * k / K_DIRS), math.sin(2 * math.pi * k / K_DIRS)]
                       for k in range(K_DIRS)], dtype=np.float32)


def set_determinism(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_critic(device):
    critic = CriticLSTM(film=True).to(device)
    critic.load_state_dict(torch.load(CRITIC_PATH, map_location=device), strict=True)
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False
    return critic


def to_vision_tensor(v01, device):
    """v01: (B, H, W, 3) float in [0,1] -> (B, 3, H, W) tensor."""
    return torch.from_numpy(np.ascontiguousarray(np.transpose(v01, (0, 3, 1, 2)))).float().to(device)


@torch.no_grad()
def _trunk(critic, v, r):
    """The action-independent part of CriticLSTM.forward (CNN, FiLM, LSTM)."""
    feat = critic.encoder(v)
    if critic.film is not None:
        feat = critic.film(feat, r)
    lstm_out, _ = critic.lstm(feat.unsqueeze(1), None)
    return lstm_out.squeeze(1)


@torch.no_grad()
def _head(critic, h, actions, r):
    """h: (B, H), actions: (B, A, 2), r: (B, 4) -> Q (B, A)."""
    b, a = actions.shape[:2]
    x = torch.cat([h.unsqueeze(1).expand(-1, a, -1), actions, r.unsqueeze(1).expand(-1, a, -1)], dim=-1)
    return critic.q(x).squeeze(-1)


def unit_actions(motion):
    """Normalize observed actions to unit length (§2.5). Returns (unit, small_mask)."""
    norm = np.linalg.norm(motion, axis=-1, keepdims=True)
    small = norm[..., 0] < SMALL_ACTION
    unit = motion / np.maximum(norm, 1e-12)
    return unit.astype(np.float32), small


@torch.no_grad()
def check_trunk_head_equivalence(critic, vision_u8, device, n=64, seed=0, atol=1e-5):
    """Compare the split computation with CriticLSTM.forward on random frames,
    random candidate r and random unit actions. Raises if they differ."""
    rng = np.random.RandomState(seed)
    flat = vision_u8.reshape(-1, *vision_u8.shape[-3:])
    idx = rng.choice(len(flat), size=n, replace=False)
    v = to_vision_tensor(flat[idx].astype(np.float32) / 255.0, device)
    r = torch.from_numpy(CAND_R[rng.randint(len(CAND_R), size=n)]).to(device)
    a = torch.from_numpy(PROBE_DIRS[rng.randint(K_DIRS, size=n)]).to(device)
    q_ref, _ = critic(v, a, r, hidden=None)
    q_split = _head(critic, _trunk(critic, v, r), a.unsqueeze(1), r)
    diff = float((q_ref.squeeze(-1) - q_split.squeeze(-1)).abs().max())
    if diff > atol:
        raise RuntimeError(f'trunk/head split differs from CriticLSTM.forward: max|diff|={diff}')
    return diff


@torch.no_grad()
def compute_q(critic, vision01_fn, n_frames, unit_obs, device, batch=2048):
    """
    vision01_fn(i0, i1) -> (i1-i0, H, W, 3) float [0,1] frames (flattened index).
    unit_obs: (n_frames, 2) unit observed actions.
    Returns Q64 (n_frames, R, 64) and Qobs (n_frames, R), float32, for all CAND_R.
    """
    n_r = len(CAND_R)
    q64 = np.zeros((n_frames, n_r, K_DIRS), dtype=np.float32)
    qobs = np.zeros((n_frames, n_r), dtype=np.float32)
    dirs = torch.from_numpy(PROBE_DIRS).to(device)
    for i0 in range(0, n_frames, batch):
        i1 = min(i0 + batch, n_frames)
        v = to_vision_tensor(vision01_fn(i0, i1), device)
        b = i1 - i0
        acts = torch.cat([dirs.unsqueeze(0).expand(b, -1, -1),
                          torch.from_numpy(unit_obs[i0:i1]).to(device).unsqueeze(1)], dim=1)
        for j in range(n_r):
            r = torch.from_numpy(CAND_R[j]).to(device).unsqueeze(0).expand(b, -1)
            q = _head(critic, _trunk(critic, v, r), acts, r).cpu().numpy()
            q64[i0:i1, j] = q[:, :K_DIRS]
            qobs[i0:i1, j] = q[:, K_DIRS]
    return q64, qobs


def frame_loglik(q64, qobs, beta):
    """(..., R, 64), (..., R) -> log pi(a_obs | s, r, beta), (..., R). float64."""
    q64 = q64.astype(np.float64)
    return beta * qobs.astype(np.float64) - logsumexp(beta * q64, axis=-1)


def log_posterior_all_t(q64, qobs, use_mask, betas, log_prior):
    """
    q64 (N, T, R, 64), qobs (N, T, R), use_mask (N, T) bool (frames that
    update the posterior), log_prior (R,).
    Returns log P(r | frames 0..t-1) for t = 0..T as (N, T+1, R); index t
    means t observed pairs (t=0 is the prior).
    """
    n, T, R = qobs.shape
    per_beta = []
    for beta in betas:
        ll = frame_loglik(q64, qobs, beta) * use_mask[..., None]
        cum = np.concatenate([np.zeros((n, 1, R)), np.cumsum(ll, axis=1)], axis=1)
        per_beta.append(cum)
    L = np.stack(per_beta, axis=-1)  # (N, T+1, R, B)
    lp = logsumexp(L, axis=-1) - np.log(len(betas)) + log_prior[None, None, :]
    return lp - logsumexp(lp, axis=-1, keepdims=True)


def argmax_with_ties(scores, seed=0):
    """scores (N, R). Exact-equality ties broken by a fresh RandomState(seed).
    Returns (pred (N,), tied (N,) bool)."""
    rng = np.random.RandomState(seed)
    mx = scores.max(axis=1, keepdims=True)
    is_max = scores == mx
    tied = is_max.sum(axis=1) > 1
    pred = np.empty(len(scores), dtype=np.int64)
    for i in range(len(scores)):
        cands = np.flatnonzero(is_max[i])
        pred[i] = cands[0] if len(cands) == 1 else rng.choice(cands)
    return pred, tied


def init_bins(init_dist):
    out = np.full(len(init_dist), -1)
    for b, (lo, hi) in enumerate(DIST_BINS):
        out[(init_dist >= lo) & (init_dist < hi)] = b
    return out


def evaluate(scores, goal, bins, post=None, extra=None):
    """scores (N, R): higher = preferred. post (N, R) probabilities or None.
    Uses the true goal for scoring only."""
    pred, tied = argmax_with_ties(scores)
    correct = pred == goal
    res = dict(acc=float(correct.mean()), tie_rate=float(tied.mean()),
               n=int(len(goal)), by_goal={}, by_init_bin={},
               confusion=np.zeros((N_MAIN, scores.shape[1]), dtype=int).tolist())
    conf = np.zeros((N_MAIN, scores.shape[1]), dtype=int)
    for g, p in zip(goal, pred):
        conf[g, p] += 1
    res['confusion'] = conf.tolist()
    if post is not None:
        pc = post[np.arange(len(goal)), goal]
        res['mean_post_correct'] = float(pc.mean())
        res['mean_post_r1'] = float(post[:, R1_INDEX].mean())
        if post.shape[1] > N_MAIN:
            res['mean_post_blue'] = float(post[:, 3].mean())
    for g in range(N_MAIN):
        m = goal == g
        d = dict(n=int(m.sum()), acc=float(correct[m].mean()))
        if post is not None:
            d['mean_post_correct'] = float(post[m, g].mean())
        res['by_goal'][CAND_NAMES[g]] = d
    for b, name in enumerate(BIN_NAMES):
        m = bins == b
        d = dict(n=int(m.sum()), acc=float(correct[m].mean()) if m.any() else None)
        if post is not None and m.any():
            d['mean_post_correct'] = float(post[m, goal[m]].mean())
        res['by_init_bin'][name] = d
    if extra:
        res.update(extra)
    return res


def color_scores(fractions, n_cand=N_MAIN):
    """fractions (N, T, 4) -> cumulative sum_t r·fraction(s_t), (N, T+1, R)."""
    s = fractions.astype(np.float64) @ CAND_R[:n_cand].T.astype(np.float64)
    n = s.shape[0]
    return np.concatenate([np.zeros((n, 1, n_cand)), np.cumsum(s, axis=1)], axis=1)


def v_slope_scores(q64, t, n_cand=N_MAIN):
    """Action-free estimate (§2.5): OLS slope over frames 0..t-1 of
    V(s, r) = max_k Q(s, a_k, r). t=1 -> all zeros (exact tie)."""
    V = q64[:, :t, :n_cand].max(axis=-1).astype(np.float64)  # (N, t, R)
    if t < 2:
        return np.zeros((V.shape[0], n_cand))
    x = np.arange(t, dtype=np.float64)
    x = x - x.mean()
    return np.einsum('t,ntr->nr', x, V - V.mean(axis=1, keepdims=True)) / (x ** 2).sum()


def landmark_fraction_array(vision01):
    """(M, H, W, 3) [0,1] -> (M, 4) fractions, plus (M,) bool 'no pixel over threshold'."""
    fr = np.stack([landmark_fractions(v) for v in vision01])
    return fr, fr.sum(axis=1) == 0


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()), sd=float(x.std()),
                q05=float(np.quantile(x, 0.05)), q25=float(np.quantile(x, 0.25)),
                median=float(np.median(x)), q75=float(np.quantile(x, 0.75)),
                q95=float(np.quantile(x, 0.95)),
                frac_pos=float((x > 0).mean()), frac_neg=float((x < 0).mean()))
