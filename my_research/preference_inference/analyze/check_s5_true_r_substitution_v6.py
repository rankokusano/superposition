"""
S5 follow-up (2026-10-04, user request): isolate whether h2->other's drop
from S4's true_r oracle (0.880, self-vision variant per Sec.37.2's finding)
to S5's actual VE'-estimated value (0.53-0.54) is because (a) r_hat is far
from the true A-2 reward (so Q_hat2 is simply computed from a different,
worse r), or (b) VE' training itself perturbed process-2's use downstream.

Since S5's config (config/exp/v6_s5_ve.yml) freezes superposition_module
and probe_critic -- ONLY value_estimator_module is in the optimizer -- the
SM and critic weights in the S5 checkpoint are bit-identical to S3's
(S5 pretrains from S3 ep200 and never updates them). So explanation (b)
is not structurally possible: nothing downstream of r_hat could have
changed independently of what value flows through it. This script
verifies that directly and empirically: load the S5 (trained VE') model,
and for the SAME frozen SM/critic, replace VE's own r_hat with the fixed
true A2_TRUE_R for every frame, recompute Q_hat2 and h2, and see if
h2->other recovers toward S4/Sec.37.2's ~0.88.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/check_s5_true_r_substitution_v6.py --exp_config v6_s5_ve --epoch 400
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
from model import util as model_util  # noqa
import util  # noqa
from my_research.preference_inference.model.rl_agent_sac_v6 import A2_TRUE_R  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
SAVE_DIR = 'data/result/v6_baseline'
T = 100
BATCH = 200


class Args:
    def __init__(self, exp_config, seed):
        self.exp_config = exp_config
        self.seed = seed


def r2(X, Y):
    reg = Ridge(alpha=1.0)
    reg.fit(X, Y)
    return float(r2_score(Y, reg.predict(X)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_config', default='v6_s5_ve')
    ap.add_argument('--epoch', type=int, default=400)
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
    modes = ['r_hat_trained', 'true_r_substituted']

    with h5py.File(args.data_h5, 'r') as f:
        n_avail = f['train/self_vision'].shape[0]
        n_use = min(args.n_episodes, n_avail)
        print(f'Loading {n_use}/{n_avail} episodes ...')
        sv_all = f['train/self_vision'][:n_use, :T]
        sp_all = f['train/self_position'][:n_use, :T]
        op_all = f['train/other_position'][:n_use, :T]

    h2_all = {m: np.zeros((n_use, T, 128), dtype=np.float32) for m in modes}

    torch.manual_seed(args.eval_seed)
    with torch.no_grad():
        for b0 in range(0, n_use, BATCH):
            b1 = min(b0 + BATCH, n_use)
            bsz = b1 - b0
            model.init_state(bsz)
            init_state_snapshot = {k: v for k, v in model.superposition_module.state.items()}
            per_mode_state = {m: init_state_snapshot for m in modes}
            r_const = torch.tensor(A2_TRUE_R, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(bsz, -1)

            for t in range(T):
                sv_np = sv_all[b0:b1, t].astype(np.float32)
                if sv_np.max() > 1.5:
                    sv_np = sv_np / 255.0
                sv_t = torch.tensor(util.scale_vision(sv_np)).permute(0, 3, 1, 2).float().to(DEVICE)
                sv_raw = (sv_t + 1) / 2
                p_mask = 0.0 if t == 0 else p_mask_vision

                sv_enc = model.self_vision_encoder_module(sv_t)
                ov_enc = model.other_vision_encoder_module(sv_t)
                q1_vec = model.compute_probe_q(sv_raw)
                sv_enc_m = model_util.mask(sv_enc, p_mask)
                ov_enc_m = model_util.mask(ov_enc, p_mask)

                r_hat = model.value_estimator_module(ov_enc)  # VE's own trained output
                q2_by_mode = {
                    'r_hat_trained': model.compute_probe_q2_from_r_hat(sv_raw, r_hat),
                    'true_r_substituted': model.compute_probe_q2_from_r_hat(sv_raw, r_const),
                }

                for mode in modes:
                    model.superposition_module.state = per_mode_state[mode]
                    ss, os_ = model.superposition_module(sv_enc_m, q1_vec, ov_enc_m, q2_by_mode[mode])
                    per_mode_state[mode] = model.superposition_module.state
                    h2_all[mode][b0:b1, t] = os_.cpu().numpy()

    sp_flat = sp_all.reshape(-1, 2)
    op_flat = op_all.reshape(-1, 2)
    result = {}
    print('\n=== h2 -> position R^2: VE-trained r_hat vs true-r substituted (SAME frozen SM/critic) ===')
    for mode in modes:
        h2_flat = h2_all[mode].reshape(-1, 128)
        r2_self = r2(h2_flat, sp_flat)
        r2_other = r2(h2_flat, op_flat)
        result[mode] = {'h2_to_self': r2_self, 'h2_to_other': r2_other}
        print(f'  {mode:20s}: h2->self={r2_self:.4f}  h2->other={r2_other:.4f}')

    print(f'\n  reference: SS37.2 self-vision true_r oracle on S3 checkpoint gave h2->other ~ (not directly '
          f'measured there; S4 SS35.2 other_vision oracle true_r gave h2->other=0.880)')

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{args.exp_config}_true_r_substitution_check.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
