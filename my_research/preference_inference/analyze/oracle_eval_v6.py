"""
v6 S4 oracle experiment (docs/v6_instructions.md Sec.3 "S4", pre-registered
design in docs/v6_experiment_log.md Sec.33, written BEFORE this script was
run). Most important gate of the whole v6 line: without training any VE,
feed A-2's TRUE reward parameter directly into the SAME frozen, reward-
conditioned critic used for process-1, applied to the REAL other_vision,
and see whether that helps self_vision reconstruction at all. If
`true_r >= zero` (matching v4/v5's P0-4 failure: true_q2 - zero = +0.00441,
i.e. the correct answer made things WORSE), the mechanism itself is
questionable and S5 (estimating r from self-vision alone, a strictly
harder problem) is not worth attempting.

Deliberately mirrors analyze/r4_ve_eval.py's ablation-by-substitution
pattern (loads a FROZEN S3 checkpoint, no new training -- there is nothing
to train here since q2 for every mode is a deterministic function of the
already-trained probe_critic) including its stateful-superposition_module
bugfix (one state snapshot per mode, swapped in/out around each mode's
call within a timestep -- without this, mode N inherits mode N-1's
mutated state and the four "conditions" are not independent trajectories).

Ablation modes for process-2's Q-hat (docs/v6_experiment_log.md Sec.33.1):
  - true_r  : other_vision + r=A2_TRUE_R   (expected best)
  - wrong_r : other_vision + r=A1_TRUE_R   (expected bad -- mismatched belief)
  - zero    : zeros (S2/S3's original process-2 input; expected middle)
  - constant: true_r's per-frame mean, replicated across all 8 directions
              (removes directional structure only, keeps state-varying level)

mu/sigma reuse process-1's q_mu/q_sigma (SS33.1: the whole point of a
conditioned critic is that the SAME normalisation transfers across r).

Also computes, per mode, TWO variants of ov_enc (SS33.1's secondary
robustness check):
  - 'other_vision_ov_enc' (primary): other_vision_encoder_module(REAL other_vision)
  - 'self_vision_ov_enc' (secondary): other_vision_encoder_module(self_vision),
    i.e. left exactly as S2/S3 trained it, only q2_vec is swapped.
If the two disagree on the true_r<zero verdict, that itself is a finding
(whether ov_enc's access to real other_vision matters) -- not a reason to
prefer one over the other after the fact.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/oracle_eval_v6.py --exp_config v6_s3_base_l1 --epoch 200
"""
import argparse
import json
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
from model import util as model_util  # noqa
import util  # noqa
from my_research.preference_inference.model.rl_agent_sac_v6 import A1_TRUE_R, A2_TRUE_R  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
SAVE_DIR = 'data/result/baseline_v4'
N_EPISODES = 3000
T = 100
BATCH = 200
MODES = ['true_r', 'wrong_r', 'zero', 'constant']
OV_ENC_SOURCES = ['other_vision', 'self_vision']


class Args:
    def __init__(self, exp_config, seed):
        self.exp_config = exp_config
        self.seed = seed


def compute_q2(model, ov_raw, mode, r_vec, bsz, K):
    if mode == 'zero':
        return torch.zeros(bsz, K, device=DEVICE)
    r = torch.tensor(r_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(bsz, -1)
    v_rep = ov_raw.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(bsz * K, *ov_raw.shape[1:])
    a_rep = model.probe_actions.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * K, 2)
    r_rep = r.unsqueeze(1).expand(-1, K, -1).reshape(bsz * K, -1)
    with torch.no_grad():
        q, _ = model.probe_critic(v_rep, a_rep, r_rep, hidden=None)
    q = q.reshape(bsz, K)
    return torch.tanh((q - model.q_mu) / model.q_sigma)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', default='v6_s3_base_l1')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES)
    parser.add_argument('--label', default=None)
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--data_h5', default=DATA_H5)
    args = parser.parse_args()
    label = args.label or f'{args.exp_config}_oracle_eval'

    import random as _random
    _random.seed(args.eval_seed); np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed); torch.cuda.manual_seed_all(args.eval_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    exp_config = util.gen_exp_config(Args(args.exp_config, args.seed))
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(args.exp_config, args.seed), test=False)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(DEVICE)
    util.load_model(model_dir, args.epoch, model)
    model.eval()

    p_mask_vision = exp_config.p_mask_vision
    K = model.probe_actions.size(0)
    r_by_mode = {'true_r': A2_TRUE_R, 'wrong_r': A1_TRUE_R, 'zero': None, 'constant': A2_TRUE_R}

    with h5py.File(args.data_h5, 'r') as f:
        n_avail = f['train/self_vision'].shape[0]
        n_use = min(args.n_episodes, n_avail)
        print(f'Loading {n_use}/{n_avail} episodes ...')
        sv_all = f['train/self_vision'][:n_use, :T]
        ov_all = f['train/other_vision'][:n_use, :T]
        sp_all = f['train/self_position'][:n_use, :T]
        op_all = f['train/other_position'][:n_use, :T]

    conditions = [(src, mode) for src in OV_ENC_SOURCES for mode in MODES]
    vision_l1 = {c: 0.0 for c in conditions}
    n_samples = 0
    # process-2 output stream ("h2"), collected for the PRIMARY (other_vision) source only,
    # to check the SS33.2 "does h2->self rise" pathology check.
    os_all = {mode: np.zeros((n_use, T, 128), dtype=np.float32) for mode in MODES}

    torch.manual_seed(args.eval_seed)
    with torch.no_grad():
        for b0 in range(0, n_use, BATCH):
            b1 = min(b0 + BATCH, n_use)
            bsz = b1 - b0
            model.init_state(bsz)
            # Bugfix from r4_ve_eval.py (2026-08-21): superposition_module is
            # stateful; each condition needs its own independent trajectory.
            init_state_snapshot = {k: v for k, v in model.superposition_module.state.items()}
            per_cond_state = {c: init_state_snapshot for c in conditions}

            for t in range(T):
                sv_np = sv_all[b0:b1, t].astype(np.float32)
                if sv_np.max() > 1.5:
                    sv_np = sv_np / 255.0
                sv_t = torch.tensor(util.scale_vision(sv_np)).permute(0, 3, 1, 2).float().to(DEVICE)
                sv_raw = (sv_t + 1) / 2

                ov_np = ov_all[b0:b1, t].astype(np.float32)
                if ov_np.max() > 1.5:
                    ov_np = ov_np / 255.0
                ov_t = torch.tensor(util.scale_vision(ov_np)).permute(0, 3, 1, 2).float().to(DEVICE)
                ov_raw = (ov_t + 1) / 2

                p_mask = 0.0 if t == 0 else p_mask_vision

                sv_enc = model.self_vision_encoder_module(sv_t)
                ov_enc_other = model.other_vision_encoder_module(ov_t)   # primary: real other_vision
                ov_enc_self = model.other_vision_encoder_module(sv_t)    # secondary: S2/S3's own convention
                q1_vec = model.compute_probe_q(sv_raw)

                sv_enc_m = model_util.mask(sv_enc, p_mask)
                ov_enc_other_m = model_util.mask(ov_enc_other, p_mask)
                ov_enc_self_m = model_util.mask(ov_enc_self, p_mask)
                ov_enc_m_by_src = {'other_vision': ov_enc_other_m, 'self_vision': ov_enc_self_m}

                q2_true_r = compute_q2(model, ov_raw, 'true_r', A2_TRUE_R, bsz, K)
                q2_by_mode = {
                    'true_r': q2_true_r,
                    'wrong_r': compute_q2(model, ov_raw, 'wrong_r', A1_TRUE_R, bsz, K),
                    'zero': compute_q2(model, ov_raw, 'zero', None, bsz, K),
                    'constant': q2_true_r.mean(dim=1, keepdim=True).expand(-1, K),
                }

                for src, mode in conditions:
                    q2_vec = q2_by_mode[mode]
                    model.superposition_module.state = per_cond_state[(src, mode)]
                    ss, os_ = model.superposition_module(sv_enc_m, q1_vec, ov_enc_m_by_src[src], q2_vec)
                    per_cond_state[(src, mode)] = model.superposition_module.state
                    so = model.integration_module(ss, os_)
                    vision_pred = model.vision_decoder_module(so)
                    vision_l1[(src, mode)] += (vision_pred - sv_t).abs().mean().item() * bsz
                    if src == 'other_vision':
                        os_all[mode][b0:b1, t] = os_.cpu().numpy()

                n_samples += bsz

    for c in conditions:
        vision_l1[c] /= n_samples

    print('\n=== S4 primary result: self_vision L1 loss by (ov_enc source, mode) ===')
    for src in OV_ENC_SOURCES:
        print(f'  ov_enc = {src}:')
        for mode in MODES:
            print(f'    {mode:10s}: {vision_l1[(src, mode)]:.4f}')
        d = vision_l1[(src, 'true_r')] - vision_l1[(src, 'zero')]
        print(f'    true_r - zero = {d:+.4f}  '
              f"({'true_r HELPS (design works)' if d < 0 else 'true_r HURTS (v4/v5-style failure)'})")

    # --- SS33.2: h2->self / h2->other R^2 per mode (primary ov_enc=other_vision only) ---
    print('\n=== h2 (process-2 output, ov_enc=other_vision) -> position R^2 by mode ===')
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    def r2(X, Y):
        reg = Ridge(alpha=1.0)
        reg.fit(X, Y)
        return r2_score(Y, reg.predict(X))

    h2_r2 = {}
    sp_flat = sp_all.reshape(-1, 2)
    op_flat = op_all.reshape(-1, 2)
    for mode in MODES:
        h2_flat = os_all[mode].reshape(-1, 128)
        r2_self = r2(h2_flat, sp_flat)
        r2_other = r2(h2_flat, op_flat)
        h2_r2[mode] = {'h2_to_self': r2_self, 'h2_to_other': r2_other}
        print(f'  {mode:10s}: h2->self={r2_self:.4f}  h2->other={r2_other:.4f}')

    result = {
        'label': label,
        'exp_config': args.exp_config,
        'epoch': args.epoch,
        'eval_seed': args.eval_seed,
        'n_episodes': n_use,
        'vision_l1_by_condition': {f'{src}__{mode}': vision_l1[(src, mode)]
                                    for src, mode in conditions},
        'true_r_minus_zero': {
            src: vision_l1[(src, 'true_r')] - vision_l1[(src, 'zero')] for src in OV_ENC_SOURCES
        },
        'h2_r2_by_mode': h2_r2,
        'note': 'primary decision uses ov_enc=other_vision (SS33.1 main design); '
                'ov_enc=self_vision is the SS33.1 secondary robustness check',
    }
    result.update(util.gen_result_metadata(
        exp_config_name=args.exp_config, seed=0,
        dataset_name=os.path.basename(os.path.dirname(args.data_h5))))

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{label}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
