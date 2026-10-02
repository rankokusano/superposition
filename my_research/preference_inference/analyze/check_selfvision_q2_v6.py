"""
S5 pre-flight check (2026-10-02): S4's oracle (SS33.1) computed q2 from
REAL other_vision + true r -- a deliberately generous ceiling, since the
real system never has other_vision access. S5's actual VE'->critic
pipeline must stay faithful to the "self-vision only" thesis: VE' reads
ov_enc derived from SELF vision (matching S2/S3's base-stage convention,
other_vision_encoder_module(self_vision)), and the resulting r_hat is fed
back into the SAME critic using SELF vision as well (not real other_vision)
-- Q_hat2 = critic(self_vision, probe_actions, r_hat), i.e. "what would MY
value be if I had the OTHER's (estimated) preferences, from MY own current
view." This is a strictly HARDER setting than what S4 tested (self-vision
instead of real other_vision for the critic's visual argument), so before
committing to an S5 training run, check whether the SS35/SS36 distance-
dependent true_r benefit survives when q2 is computed this way.

Reuses oracle_eval_v6.py's run_epoch() machinery conceptually but with q2
computed from SELF vision (not other_vision) for the true_r/zero modes
only (this is a quick go/no-go check, not the full 4-mode/2-source battery
already done for S4).

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/check_selfvision_q2_v6.py --exp_config v6_s3_base_l1 --epoch 200
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
from my_research.preference_inference.model.rl_agent_sac_v6 import A2_TRUE_R  # noqa
from oracle_eval_v6 import compute_q2, Args, GREEN_POS, BIN_EDGES, BIN_LABELS, DATA_H5, T, BATCH  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'data/result/v6_baseline'
MODES = ['true_r', 'zero']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', default='v6_s3_base_l1')
    ap.add_argument('--epoch', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_episodes', type=int, default=3000)
    ap.add_argument('--eval_seed', type=int, default=0)
    ap.add_argument('--data_h5', default=DATA_H5)
    args = ap.parse_args()

    import random as _random
    _random.seed(args.eval_seed); np.random.seed(args.eval_seed)
    torch.manual_seed(args.eval_seed); torch.cuda.manual_seed_all(args.eval_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    exp_config = util.gen_exp_config(Args(args.exp_config, args.seed))
    model_config = util.gen_model_config(exp_config)
    _, model_dir, _ = util.gen_dirs(Args(args.exp_config, args.seed), test=False)
    model = getattr(models, exp_config.model.name)(model_config)
    model.to(DEVICE)
    util.load_model(model_dir, args.epoch, model)
    model.eval()

    p_mask_vision = exp_config.p_mask_vision
    K = model.probe_actions.size(0)

    with h5py.File(args.data_h5, 'r') as f:
        n_avail = f['train/self_vision'].shape[0]
        n_use = min(args.n_episodes, n_avail)
        print(f'Loading {n_use}/{n_avail} episodes ...')
        sv_all = f['train/self_vision'][:n_use, :T]
        op_all = f['train/other_position'][:n_use, :T]

    vision_l1 = {m: 0.0 for m in MODES}
    vision_l1_frames = {m: np.zeros((n_use, T), dtype=np.float32) for m in MODES}
    n_samples = 0

    torch.manual_seed(args.eval_seed)
    with torch.no_grad():
        for b0 in range(0, n_use, BATCH):
            b1 = min(b0 + BATCH, n_use)
            bsz = b1 - b0
            model.init_state(bsz)
            init_state_snapshot = {k: v for k, v in model.superposition_module.state.items()}
            per_mode_state = {m: init_state_snapshot for m in MODES}

            for t in range(T):
                sv_np = sv_all[b0:b1, t].astype(np.float32)
                if sv_np.max() > 1.5:
                    sv_np = sv_np / 255.0
                sv_t = torch.tensor(util.scale_vision(sv_np)).permute(0, 3, 1, 2).float().to(DEVICE)
                sv_raw = (sv_t + 1) / 2
                p_mask = 0.0 if t == 0 else p_mask_vision

                sv_enc = model.self_vision_encoder_module(sv_t)
                ov_enc = model.other_vision_encoder_module(sv_t)  # S5's real convention: self-vision
                q1_vec = model.compute_probe_q(sv_raw)
                sv_enc_m = model_util.mask(sv_enc, p_mask)
                ov_enc_m = model_util.mask(ov_enc, p_mask)

                # KEY DIFFERENCE from SS33.1's oracle: q2 computed from SELF
                # vision (sv_raw), not other_vision -- matching what S5's
                # VE'->critic pipeline will actually do.
                q2_by_mode = {
                    'true_r': compute_q2(model, sv_raw, 'true_r', A2_TRUE_R, bsz, K),
                    'zero': compute_q2(model, sv_raw, 'zero', None, bsz, K),
                }

                for mode in MODES:
                    model.superposition_module.state = per_mode_state[mode]
                    ss, os_ = model.superposition_module(sv_enc_m, q1_vec, ov_enc_m, q2_by_mode[mode])
                    per_mode_state[mode] = model.superposition_module.state
                    so = model.integration_module(ss, os_)
                    vision_pred = model.vision_decoder_module(so)
                    per_sample_l1 = (vision_pred - sv_t).abs().mean(dim=(1, 2, 3))
                    vision_l1[mode] += per_sample_l1.sum().item()
                    vision_l1_frames[mode][b0:b1, t] = per_sample_l1.cpu().numpy()

                n_samples += bsz

    for m in MODES:
        vision_l1[m] /= n_samples
    print(f'\n=== self-vision-q2 check: true_r vs zero (q2 computed from SELF vision) ===')
    for m in MODES:
        print(f'  {m:10s}: {vision_l1[m]:.4f}')
    d = vision_l1['true_r'] - vision_l1['zero']
    print(f'  true_r - zero = {d:+.4f}  ({"helps" if d < 0 else "hurts"})')

    op_flat = op_all.reshape(-1, 2)
    dist_flat = np.linalg.norm(GREEN_POS[None, :] - op_flat, axis=1)
    print('\n=== distance-binned (self-vision q2) ===')
    dist_bins = {}
    for lo, hi, lbl in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_LABELS):
        m_idx = (dist_flat >= lo) & (dist_flat < hi)
        n = int(m_idx.sum())
        row = {mode: float(vision_l1_frames[mode].reshape(-1)[m_idx].mean()) for mode in MODES}
        delta = row['true_r'] - row['zero']
        dist_bins[lbl] = {'n': n, **row, 'true_r_minus_zero': delta}
        print(f'  {lbl:6s} (n={n:6d}): true_r={row["true_r"]:.4f}  zero={row["zero"]:.4f}  '
              f'delta={delta:+.4f}')

    result = {
        'exp_config': args.exp_config, 'epoch': args.epoch, 'n_episodes': n_use,
        'note': 'q2 computed from SELF vision (S5s real convention), not other_vision (SS33.1 oracle)',
        'vision_l1': vision_l1,
        'true_r_minus_zero_overall': vision_l1['true_r'] - vision_l1['zero'],
        'distance_binned': dist_bins,
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{args.exp_config}_selfvision_q2_check.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
