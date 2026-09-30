"""
S2 follow-up diagnostic (2026-09-30, docs/v6_experiment_log.md SS31):
why is h1->self's in-distribution -> canonical drop larger for v6 than v5
(v6: 0.82 -> 0.49, v5: 0.80 -> 0.72)?

Candidate (b): the mu/sigma normalisation in SuperpositionNetworkProbeQV6
(config/model/SuperpositionNetworkProbeQV6/default.yml) was calibrated
SOLELY on r3_stay self_vision frames (docs/v6_experiment_log.md SS29.2).
If the FiLM critic's raw (pre-tanh) Q distribution on r2_a1random_a2rl
frames has a different mean/spread than on r3_stay, the r3_stay-calibrated
(q-mu)/sigma will saturate the tanh on r2_a1random_a2rl, destroying the
per-direction information the probe-Q vector is supposed to carry into
process-1 -- a distribution-shift/mis-calibration effect, not a generic
"sigma too small" effect (sigma was tuned to be fine ON r3_stay by
construction).

This script loads the v6 S2 checkpoint (which owns model.probe_critic,
model.probe_actions, model.probe_r, model.q_mu, model.q_sigma) and
recomputes the RAW (pre-tanh) probe-Q values for self_vision frames drawn
from both r3_stay and r2_a1random_a2rl, then reports:
  - raw Q mean/std per dataset (vs. the calibration mu/sigma)
  - saturation fraction: P(|tanh((q-mu)/sigma)| > 0.95) per dataset
  - per-direction (8-way) std of the NORMALISED q vector per dataset
    (this is what actually carries direction information into process-1;
    if it collapses on r2_a1random_a2rl, that's the mechanism for the
    weaker h1->self there)

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/diag_probeq_calibration_v6.py --exp_config v6_s2_base_mse --epoch 400
"""
import argparse
import os
import sys

import h5py
import numpy as np
import torch

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if '/work' not in sys.path:
    sys.path.insert(0, '/work')
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import model as models  # noqa
import util  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
T = 100
BATCH = 200
N_EPISODES = 1000
DATASETS = {
    'r3_stay': '/work/my_research/preference_inference/data/data/r3_stay/data.h5',
    'r2_a1random_a2rl': '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5',
}


class Args:
    def __init__(self, exp_config, seed):
        self.exp_config = exp_config
        self.seed = seed


def load_self_vision(h5_path, n_episodes):
    with h5py.File(h5_path, 'r') as f:
        n_avail = f['train/self_vision'].shape[0]
        n_use = min(n_episodes, n_avail)
        return f['train/self_vision'][:n_use, :T]


def raw_and_norm_q(model, sv_all):
    n_use = sv_all.shape[0]
    k = model.probe_actions.size(0)
    raw_all = np.zeros((n_use, T, k), dtype=np.float32)
    norm_all = np.zeros((n_use, T, k), dtype=np.float32)
    with torch.no_grad():
        for b0 in range(0, n_use, BATCH):
            b1 = min(b0 + BATCH, n_use)
            for t in range(T):
                sv_np = sv_all[b0:b1, t].astype(np.float32)
                if sv_np.max() > 1.5:
                    sv_np = sv_np / 255.0
                sv_scaled = util.scale_vision(sv_np)
                sv_t = torch.tensor(sv_scaled).permute(0, 3, 1, 2).float().to(DEVICE)
                sv_raw = (sv_t + 1) / 2

                b = sv_raw.size(0)
                v_rep = sv_raw.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(b * k, *sv_raw.shape[1:])
                a_rep = model.probe_actions.unsqueeze(0).expand(b, -1, -1).reshape(b * k, 2)
                r_rep = model.probe_r.unsqueeze(0).expand(b * k, -1)
                q, _ = model.probe_critic(v_rep, a_rep, r_rep, hidden=None)
                q = q.reshape(b, k)
                q_norm = torch.tanh((q - model.q_mu) / model.q_sigma)

                raw_all[b0:b1, t] = q.cpu().numpy()
                norm_all[b0:b1, t] = q_norm.cpu().numpy()
    return raw_all, norm_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', default='v6_s2_base_mse')
    ap.add_argument('--epoch', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_episodes', type=int, default=N_EPISODES)
    args = ap.parse_args()

    exp_config = util.gen_exp_config(Args(args.exp_config, args.seed))
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(args.exp_config, args.seed), test=False)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(DEVICE)
    util.load_model(model_dir, args.epoch, model)
    model.eval()

    print(f'calibration mu={float(model.q_mu):.4f}  sigma={float(model.q_sigma):.4f}\n')

    for name, path in DATASETS.items():
        sv_all = load_self_vision(path, args.n_episodes)
        raw_all, norm_all = raw_and_norm_q(model, sv_all)

        raw_flat = raw_all.reshape(-1)
        norm_flat = norm_all.reshape(-1)
        sat_frac = float(np.mean(np.abs(norm_flat) > 0.95))

        # per-direction std at each (episode,t), then averaged -- this is the
        # "D" term of the direction-information metric (probe_q_direction_info_v6.py)
        per_frame_dir_std = norm_all.std(axis=-1)  # (n_episodes, T)

        print(f'--- {name} (n_episodes={sv_all.shape[0]}) ---')
        print(f'  raw Q:      mean={raw_flat.mean():.4f}  std={raw_flat.std():.4f}  '
              f'p1={np.percentile(raw_flat,1):.4f}  p99={np.percentile(raw_flat,99):.4f}')
        print(f'  norm Q:     mean={norm_flat.mean():.4f}  std={norm_flat.std():.4f}')
        print(f'  saturation: P(|norm Q| > 0.95) = {sat_frac:.4f}')
        print(f'  per-frame direction std (D-term): mean={per_frame_dir_std.mean():.4f}  '
              f'std={per_frame_dir_std.std():.4f}')
        print()


if __name__ == '__main__':
    main()
