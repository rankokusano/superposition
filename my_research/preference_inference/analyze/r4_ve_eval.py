"""
R4 VE evaluation (2026-08-21), per v4_experiment_log.md Sec.8.2 / the
2026-08-21 correction to the R4 blocker report.

VE's training signal is vision-prediction loss ONLY -- it never sees
A-2's true Q-value. So the evaluation questions are NOT "does Q̂2 match
the true Q value" (that correlation is expected to be low-sensitivity
given A-2's true Q variance is tiny -- see
docs/r4_a2_q_variance_blocker_report.md) but:

  1. Does Q̂2 actually contribute to the vision-prediction task at all?
     Direct ablation: replace Q̂2 with a fixed zero vector (R3-A's own
     original process-2 input) and with each timestep's own mean
     (removes directional structure only, keeps state-varying level) --
     compare self_vision L1 reconstruction loss against the real Q̂2.
  2. Q̂2's own statistics (mean/std/range) and a spatial heatmap against
     A-2's real position.
  3. Correlation with A-2's true Q-value (reference-only, low expected
     sensitivity given the ~0.02% variance ratio -- recorded anyway).
  4. Directional PATTERN match: does Q̂2's highest-value probe direction
     point toward Green from A-2's actual position? This is the
     evaluation that doesn't depend on absolute-scale agreement, only on
     whether the right direction is favored -- the analogous check to
     how the original paper judged MG (did the generated motion resemble
     the real motion's pattern, not exact magnitude).

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/r4_ve_eval.py --exp_config r4_ve --epoch 200
"""
import argparse
import json
import math
import os
import sys

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if '/work' not in sys.path:
    sys.path.insert(0, '/work')
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

import model as models  # noqa
from model import util as model_util  # noqa
import util  # noqa
from my_research.rl_agent_sac import CriticLSTM  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
A2_CRITIC_PATH = '/work/my_research/preference_inference/data/model/v3_rl_a2_critic.pth'
A1_MU, A1_SIGMA = 1.9233185578310863, 0.9465510185824275
GREEN_POS = np.array([-9.0, -9.0])
SAVE_DIR = 'data/result/baseline_v4'
N_EPISODES = 3000
T = 100
BATCH = 200


class Args:
    def __init__(self, exp_config, seed):
        self.exp_config = exp_config
        self.seed = seed


def apply_ablation(q2_vec, mode, true_q2_vec=None):
    if mode == 'real':
        return q2_vec
    elif mode == 'zero':
        return torch.zeros_like(q2_vec)
    elif mode == 'constant_mean':
        m = q2_vec.mean(dim=1, keepdim=True)
        return m.expand_as(q2_vec)
    elif mode == 'true_q2':
        # 2026-08-21 addition: substitute A-2's ACTUAL true Q (tanh-normalized
        # with A-1's shared stats, std~0.0092 -- the real narrow-range value)
        # in place of VE's output, to test whether SM's frozen weights
        # penalize this out-of-VE-distribution-but-true input.
        return true_q2_vec
    else:
        raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', default='r4_ve')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES)
    parser.add_argument('--label', default=None)
    args = parser.parse_args()
    label = args.label or f'{args.exp_config}_eval'

    exp_config = util.gen_exp_config(Args(args.exp_config, 0))
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(args.exp_config, 0), test=False)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(DEVICE)
    util.load_model(model_dir, args.epoch, model)
    model.eval()

    p_mask_vision = exp_config.p_mask_vision

    a2_critic = CriticLSTM()
    a2_critic.load_state_dict(torch.load(A2_CRITIC_PATH, map_location=DEVICE))
    a2_critic.to(DEVICE)
    a2_critic.eval()
    for p in a2_critic.parameters():
        p.requires_grad = False

    K = model.probe_actions.size(0)
    probe_actions_np = model.probe_actions.cpu().numpy()  # (K,2) unit vectors

    with h5py.File(DATA_H5, 'r') as f:
        n_avail = f['train/self_vision'].shape[0]
        n_use = min(args.n_episodes, n_avail)
        print(f'Loading {n_use}/{n_avail} episodes ...')
        sv_all = f['train/self_vision'][:n_use, :T]
        ov_all = f['train/other_vision'][:n_use, :T]
        op_all = f['train/other_position'][:n_use, :T]

    modes = ['real', 'zero', 'constant_mean', 'true_q2']
    vision_l1 = {m: 0.0 for m in modes}
    n_samples = 0

    q2_all = np.zeros((n_use, T, K), dtype=np.float32)
    true_q2_all = np.zeros((n_use, T, K), dtype=np.float32)

    torch.manual_seed(0)
    with torch.no_grad():
        for b0 in range(0, n_use, BATCH):
            b1 = min(b0 + BATCH, n_use)
            bsz = b1 - b0
            model.init_state(bsz)
            # BUG FIX (2026-08-21): superposition_module is stateful and its
            # forward() overwrites self.state every call. Calling it once
            # per ablation mode inside the same timestep, sharing one
            # model instance, means each mode after the first silently
            # inherits the PREVIOUS mode's mutated state instead of its own
            # -- the four conditions were not actually independent
            # trajectories. Fix: keep one state snapshot per mode and swap
            # it in/out of model.superposition_module.state around each
            # mode's call.
            init_state_snapshot = {
                k: v for k, v in model.superposition_module.state.items()
            }
            per_mode_state = {mode: init_state_snapshot for mode in modes}
            for t in range(T):
                sv_np = sv_all[b0:b1, t].astype(np.float32)
                if sv_np.max() > 1.5:
                    sv_np = sv_np / 255.0
                sv_t = torch.tensor(util.scale_vision(sv_np)).permute(0, 3, 1, 2).float().to(DEVICE)

                ov_np = ov_all[b0:b1, t].astype(np.float32)
                if ov_np.max() > 1.5:
                    ov_np = ov_np / 255.0
                ov_t = torch.tensor(util.scale_vision(ov_np)).permute(0, 3, 1, 2).float().to(DEVICE)
                ov_raw = (ov_t + 1) / 2

                p_mask = 0.0 if t == 0 else p_mask_vision

                sv_enc = model.self_vision_encoder_module(sv_t)
                ov_enc = model.other_vision_encoder_module(sv_t)
                sv_raw = (sv_t + 1) / 2
                q1_vec = model.compute_probe_q(sv_raw)
                q2_vec_real = model.value_estimator_module(ov_enc)

                # true A-2 Q, from A-2's own real other_vision + own critic
                v_rep = ov_raw.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(
                    bsz * K, *ov_raw.shape[1:])
                a_rep = model.probe_actions.unsqueeze(0).expand(bsz, -1, -1).reshape(bsz * K, 2)
                q2_true_raw, _ = a2_critic(v_rep, a_rep, hidden=None)
                q2_true = torch.tanh((q2_true_raw.reshape(bsz, K) - A1_MU) / A1_SIGMA)

                q2_all[b0:b1, t] = q2_vec_real.cpu().numpy()
                true_q2_all[b0:b1, t] = q2_true.cpu().numpy()

                sv_enc_m = model_util.mask(sv_enc, p_mask)
                ov_enc_m = model_util.mask(ov_enc, p_mask)

                for mode in modes:
                    q2_vec = apply_ablation(q2_vec_real, mode, true_q2_vec=q2_true)
                    model.superposition_module.state = per_mode_state[mode]
                    ss, os_ = model.superposition_module(sv_enc_m, q1_vec, ov_enc_m, q2_vec)
                    per_mode_state[mode] = model.superposition_module.state
                    so = model.integration_module(ss, os_)
                    vision_pred = model.vision_decoder_module(so)
                    vision_l1[mode] += (vision_pred - sv_t).abs().mean().item() * bsz

                n_samples += bsz

    for mode in modes:
        vision_l1[mode] /= n_samples

    # --- item 4: direction-pattern match against Green ---
    op_flat = op_all[:, :T].reshape(-1, 2)
    to_green = GREEN_POS[None, :] - op_flat  # (N*T, 2)
    to_green_norm = to_green / (np.linalg.norm(to_green, axis=1, keepdims=True) + 1e-8)
    q2_flat = q2_all.reshape(-1, K)
    argmax_idx = q2_flat.argmax(axis=1)
    argmax_dir = probe_actions_np[argmax_idx]  # (N*T, 2) unit vectors
    cos_sim = (argmax_dir * to_green_norm).sum(axis=1)
    angle_deg = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))

    # random-baseline expectation: if argmax were uniform over 8 directions,
    # mean cos_sim with a fixed reference direction would be 0 (symmetric)
    print(f"\n=== Item 4: does Q̂2's peak direction point toward Green? ===")
    print(f'  mean cos_sim(argmax_dir, true_dir_to_Green) = {cos_sim.mean():.4f}  '
          f'(1.0=perfect, 0.0=chance/orthogonal, -1.0=opposite)')
    print(f'  mean angular error = {angle_deg.mean():.2f} deg  '
          f'(chance expectation for uniform random 8-way pick ~ 90-101 deg)')
    print(f'  fraction within 45deg of true Green direction: '
          f'{(angle_deg < 45).mean()*100:.2f}%  (chance ~ 2/8=25%)')

    # --- item 2: Q̂2 statistics + spatial heatmap ---
    print(f"\n=== Item 2: Q̂2 statistics ===")
    print(f'  per-dim std: {q2_flat.std(axis=0)}')
    print(f'  overall: mean={q2_flat.mean():.4f} std={q2_flat.std():.4f} '
          f'range=[{q2_flat.min():.4f},{q2_flat.max():.4f}]')

    # heatmap: mean cos_sim-with-Green-direction, binned by A-2 position
    bins = 15
    xedges = np.linspace(-9.5, 9.5, bins + 1)
    yedges = np.linspace(-9.5, 9.5, bins + 1)
    xi = np.clip(np.digitize(op_flat[:, 0], xedges) - 1, 0, bins - 1)
    yi = np.clip(np.digitize(op_flat[:, 1], yedges) - 1, 0, bins - 1)
    heat_sum = np.zeros((bins, bins))
    heat_cnt = np.zeros((bins, bins))
    for i in range(len(cos_sim)):
        heat_sum[yi[i], xi[i]] += cos_sim[i]
        heat_cnt[yi[i], xi[i]] += 1
    heat = np.divide(heat_sum, heat_cnt, out=np.full_like(heat_sum, np.nan), where=heat_cnt > 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heat, origin='lower', extent=[-9.5, 9.5, -9.5, 9.5],
                    cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='cos_sim(Q̂2 argmax dir, true dir to Green)')
    ax.plot(-9, -9, 'g*', markersize=15, label='Green')
    ax.plot(-9, 9, 'r*', markersize=15, label='Red')
    ax.set_xlabel('A-2 x'); ax.set_ylabel('A-2 y')
    ax.set_title("Q̂2's peak-direction alignment with true Green direction, by A-2 position")
    ax.legend()
    plt.tight_layout()
    heatmap_path = os.path.join(SAVE_DIR, f'{label}_direction_heatmap.png')
    plt.savefig(heatmap_path, dpi=130)
    plt.close()
    print(f'  Saved heatmap: {heatmap_path}')

    # --- item 3: correlation with true A-2 Q (reference only) ---
    true_q2_flat = true_q2_all.reshape(-1, K)
    per_dim_corr = [float(np.corrcoef(q2_flat[:, k], true_q2_flat[:, k])[0, 1]) for k in range(K)]
    overall_corr = float(np.corrcoef(q2_flat.flatten(), true_q2_flat.flatten())[0, 1])
    print(f"\n=== Item 3: correlation with A-2's TRUE Q (reference only, low sensitivity expected) ===")
    print(f'  per-dim correlation: {per_dim_corr}')
    print(f'  overall (flattened) correlation: {overall_corr:.4f}')
    print(f"  true Q2 std (for context): {true_q2_flat.std():.4f}  Q̂2 std: {q2_flat.std():.4f}")

    # --- item 1: ablation vision-loss comparison ---
    print(f"\n=== Item 1: self_vision L1 loss, real Q̂2 vs ablated ===")
    for mode in modes:
        print(f'  {mode:15s}: {vision_l1[mode]:.4f}')
    print(f"  real vs zero:           delta={vision_l1['real']-vision_l1['zero']:+.4f}  "
          f"({'Q̂2 helps' if vision_l1['real'] < vision_l1['zero'] else 'Q̂2 does not help'})")
    print(f"  real vs constant_mean:  delta={vision_l1['real']-vision_l1['constant_mean']:+.4f}  "
          f"({'directional structure helps' if vision_l1['real'] < vision_l1['constant_mean'] else 'directional structure does not help'})")

    result = {
        'label': label,
        'exp_config': args.exp_config,
        'epoch': args.epoch,
        'n_episodes': n_use,
        'vision_l1_by_ablation': vision_l1,
        'q2_stats': {
            'mean': float(q2_flat.mean()), 'std': float(q2_flat.std()),
            'min': float(q2_flat.min()), 'max': float(q2_flat.max()),
            'per_dim_std': q2_flat.std(axis=0).tolist(),
        },
        'direction_match': {
            'mean_cos_sim_with_green_dir': float(cos_sim.mean()),
            'mean_angle_error_deg': float(angle_deg.mean()),
            'frac_within_45deg': float((angle_deg < 45).mean()),
        },
        'true_q_correlation': {
            'per_dim': per_dim_corr,
            'overall': overall_corr,
            'true_q2_std': float(true_q2_flat.std()),
        },
    }
    result.update(util.gen_result_metadata(
        exp_config_name=args.exp_config, seed=0, dataset_name='r2_a1random_a2rl'))

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{label}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
