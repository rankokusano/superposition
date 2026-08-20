"""
R4 pre-implementation checks (2026-08-15), per v4_experiment_log.md Sec.8.2
items 2 and 3's empirical prerequisite: before building VE, confirm there
is enough signal in the actual target (A-2's tanh-normalized probe-Q
vector) for a predictor to have any chance of learning it, and get a
rough read on whether ov_enc or h2 carries more of that signal (context
for interpreting the ov_enc-first training the user has already decided
on -- NOT a substitute for actually training VE both ways).

Two things measured, both on r2_a1random_a2rl (has real other_vision, so
A-2's OWN true state is available):

1. Target variance check: compute A-2's true K=8 probe-Q vector from A-2's
   own real other_vision (using A-2's canonical critic,
   v3_rl_a2_critic.pth), normalized with A-1's shared (mu, sigma) via
   tanh -- exactly the target VE would be trained against. Report
   per-dimension std/range, compared to A-1's own q1_vec for scale.

2. Oracle feasibility regression: Ridge-regress this target (8-dim) onto
   (a) ov_enc (other_vision run through R3-A's frozen
   other_vision_encoder_module, 64-dim) and (b) h2 (R3-A's process-2
   hidden state on the real trajectory, 128-dim) separately. This is an
   upper-bound sanity check only (linear, full-batch, no recurrence/
   causality constraints VE would actually have) -- if even this fails to
   explain much of the target's variance, that's an early warning sign
   before investing in a full VE architecture + training run.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/r4_pre_check.py --exp_config r3_a_direct --epoch 200
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if '/work' not in sys.path:
    sys.path.insert(0, '/work')
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import model as models  # noqa
import util  # noqa
from my_research.rl_agent_sac import CriticLSTM  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
A2_CRITIC_PATH = '/work/my_research/preference_inference/data/model/v3_rl_a2_critic.pth'
SAVE_DIR = 'data/result/baseline_v4'
T = 100
BATCH = 200


class Args:
    def __init__(self, exp_config, seed):
        self.exp_config = exp_config
        self.seed = seed


def r2(X, Y):
    reg = Ridge(alpha=1.0)
    reg.fit(X, Y)
    pred = reg.predict(X)
    return r2_score(Y, pred, multioutput='raw_values')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', default='r3_a_direct')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--n_episodes', type=int, default=3000)
    parser.add_argument('--label', default='r4_pre_check')
    args = parser.parse_args()

    exp_config = util.gen_exp_config(Args(args.exp_config, 0))
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(args.exp_config, 0), test=False)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(DEVICE)
    util.load_model(model_dir, args.epoch, model)
    model.eval()

    a2_critic = CriticLSTM()
    a2_critic.load_state_dict(torch.load(A2_CRITIC_PATH, map_location=DEVICE))
    a2_critic.to(DEVICE)
    a2_critic.eval()
    for p in a2_critic.parameters():
        p.requires_grad = False

    with h5py.File(DATA_H5, 'r') as f:
        n_avail = f['train/self_vision'].shape[0]
        n_use = min(args.n_episodes, n_avail)
        print(f'Loading {n_use}/{n_avail} episodes ...')
        sv_all = f['train/self_vision'][:n_use, :T]
        ov_all = f['train/other_vision'][:n_use, :T]

    K = model.probe_actions.size(0)
    q1_all = np.zeros((n_use, T, K), dtype=np.float32)   # A-1's own (reference)
    q2_all = np.zeros((n_use, T, K), dtype=np.float32)   # A-2's true, shared-normalized
    ov_enc_all = np.zeros((n_use, T, model_config.vision_encoder_module.fc[-1].output),
                           dtype=np.float32)
    h2_all = np.zeros((n_use, T, model_config.superposition_module.hidden), dtype=np.float32)

    torch.manual_seed(0)
    with torch.no_grad():
        for b0 in range(0, n_use, BATCH):
            b1 = min(b0 + BATCH, n_use)
            bsz = b1 - b0
            model.init_state(bsz)
            for t in range(T):
                sv_np = sv_all[b0:b1, t].astype(np.float32)
                if sv_np.max() > 1.5:
                    sv_np = sv_np / 255.0
                ov_np = ov_all[b0:b1, t].astype(np.float32)
                if ov_np.max() > 1.5:
                    ov_np = ov_np / 255.0

                sv_t = torch.tensor(util.scale_vision(sv_np)).permute(0, 3, 1, 2).float().to(DEVICE)
                ov_t = torch.tensor(util.scale_vision(ov_np)).permute(0, 3, 1, 2).float().to(DEVICE)

                sv_raw = (sv_t + 1) / 2
                ov_raw = (ov_t + 1) / 2

                q1_vec = model.compute_probe_q(sv_raw)  # A-1's own critic, A-1's own view

                # A-2's TRUE probe-Q, evaluated on A-2's own real view, via A-2's own critic
                b = ov_raw.size(0)
                v_rep = ov_raw.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(
                    b * K, *ov_raw.shape[1:])
                a_rep = model.probe_actions.unsqueeze(0).expand(b, -1, -1).reshape(b * K, 2)
                q2_raw, _ = a2_critic(v_rep, a_rep, hidden=None)
                q2_raw = q2_raw.reshape(b, K)
                q2_vec = torch.tanh((q2_raw - model.q_mu) / model.q_sigma)  # shared A-1 stats

                ov_enc = model.other_vision_encoder_module(sv_t)  # R3-A's own arch: this
                # branch actually encodes self_vision in R3-A/exp1_l1 (both process branches
                # read the same camera frame; there is no separate "other camera" input in
                # this architecture) -- kept for completeness/architecture-fidelity, but the
                # ov_enc->target regression below is expected to be uninformative for exactly
                # this reason. See report text.

                pred = model(
                    {'self_vision': sv_t},
                    p_mask_vision_self=exp_config.p_mask_vision,
                    p_mask_vision_other=exp_config.p_mask_vision)
                state = model.get_state()
                h2 = state['other'][0]

                q1_all[b0:b1, t] = q1_vec.cpu().numpy()
                q2_all[b0:b1, t] = q2_vec.cpu().numpy()
                ov_enc_all[b0:b1, t] = ov_enc.cpu().numpy()
                h2_all[b0:b1, t] = h2.cpu().numpy()

    q1_flat = q1_all.reshape(-1, K)
    q2_flat = q2_all.reshape(-1, K)
    ov_enc_flat = ov_enc_all.reshape(n_use * T, -1)
    h2_flat = h2_all.reshape(n_use * T, -1)

    target_stats = {
        'a1_q1_per_dim_std': q1_flat.std(axis=0).tolist(),
        'a1_q1_overall_std': float(q1_flat.std()),
        'a2_q2_per_dim_std': q2_flat.std(axis=0).tolist(),
        'a2_q2_overall_std': float(q2_flat.std()),
        'a2_q2_range': [float(q2_flat.min()), float(q2_flat.max())],
        'variance_ratio_a2_over_a1': float((q2_flat.std() / q1_flat.std()) ** 2),
    }

    r2_ov_enc = r2(ov_enc_flat, q2_flat)
    r2_h2 = r2(h2_flat, q2_flat)

    result = {
        'label': args.label,
        'exp_config': args.exp_config,
        'epoch': args.epoch,
        'n_episodes': n_use,
        'n_steps': T,
        'note': 'oracle linear (Ridge) feasibility check only -- not a trained VE, '
                'no recurrence/causality constraint. Answers "is there enough signal", '
                'not "can VE as designed learn it".',
        'target_stats': target_stats,
        'oracle_r2_ov_enc_to_a2_q2_per_dim': r2_ov_enc.tolist(),
        'oracle_r2_ov_enc_to_a2_q2_mean': float(r2_ov_enc.mean()),
        'oracle_r2_h2_to_a2_q2_per_dim': r2_h2.tolist(),
        'oracle_r2_h2_to_a2_q2_mean': float(r2_h2.mean()),
    }
    result.update(util.gen_result_metadata(
        exp_config_name=args.exp_config, seed=0, dataset_name='r2_a1random_a2rl'))

    print(f"\nA-1 q1 std (reference)        : {target_stats['a1_q1_overall_std']:.4f}")
    print(f"A-2 q2 std (VE target)        : {target_stats['a2_q2_overall_std']:.4f}  "
          f"(variance ratio vs A-1: {target_stats['variance_ratio_a2_over_a1']*100:.2f}%)")
    print(f"A-2 q2 range                  : {target_stats['a2_q2_range']}")
    print(f"\noracle R^2  ov_enc -> A-2 q2 (mean over 8 dims): {r2_ov_enc.mean():.4f}")
    print(f"oracle R^2  h2     -> A-2 q2 (mean over 8 dims): {r2_h2.mean():.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{args.label}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
