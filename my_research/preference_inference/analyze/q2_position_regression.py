"""
Q̂2 (VE output) -> position Ridge regression, at a given checkpoint epoch.

Regresses VE's raw K=8 probe-Q output against BOTH self_position (A-1) and
other_position (A-2), single-timestep, no recurrence -- answers "whose
position does Q̂2 actually encode" at a specific epoch. Used to check
whether R4-move's training-curve oscillation (docs/v4_experiment_log.md
Sec.8.x) changes this qualitative conclusion across checkpoints, or whether
it is stable regardless of which point in the oscillation is sampled.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/q2_position_regression.py --exp_config r4_move --epoch 270
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

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
SAVE_DIR = 'data/result/baseline_v4'
N_EPISODES = 3000
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
    return r2_score(Y, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', default='r4_move')
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)  # training seed of the checkpoint
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES)
    parser.add_argument('--label', default=None)
    parser.add_argument('--eval_seed', type=int, default=0)  # P0-0: reproducible eval
    args = parser.parse_args()
    label = args.label or f'{args.exp_config}_ep{args.epoch}_q2pos'

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

    with h5py.File(DATA_H5, 'r') as f:
        n_avail = f['train/self_vision'].shape[0]
        n_use = min(args.n_episodes, n_avail)
        print(f'Loading {n_use}/{n_avail} episodes ...')
        sv_all = f['train/self_vision'][:n_use, :T]
        sp_all = f['train/self_position'][:n_use, :T]
        op_all = f['train/other_position'][:n_use, :T]

    K = model.probe_actions.size(0)
    q2_all = np.zeros((n_use, T, K), dtype=np.float32)

    torch.manual_seed(args.eval_seed)
    with torch.no_grad():
        for b0 in range(0, n_use, BATCH):
            b1 = min(b0 + BATCH, n_use)
            bsz = b1 - b0
            model.init_state(bsz)
            for t in range(T):
                sv_np = sv_all[b0:b1, t].astype(np.float32)
                if sv_np.max() > 1.5:
                    sv_np = sv_np / 255.0
                sv_t = torch.tensor(util.scale_vision(sv_np)).permute(0, 3, 1, 2).float().to(DEVICE)
                ov_enc = model.other_vision_encoder_module(sv_t)
                q2_vec = model.value_estimator_module(ov_enc)
                q2_all[b0:b1, t] = q2_vec.cpu().numpy()

    q2_flat = q2_all.reshape(-1, K)
    sp_flat = sp_all.reshape(-1, 2)
    op_flat = op_all.reshape(-1, 2)

    r2_q2_self = r2(q2_flat, sp_flat)
    r2_q2_other = r2(q2_flat, op_flat)

    txt = (
        f'=== {label} (epoch {args.epoch}) ===\n'
        f'n_samples: {q2_flat.shape[0]}\n'
        f'Q̂2 -> A-1 self  pos R^2: {r2_q2_self:.4f}\n'
        f'Q̂2 -> A-2 other pos R^2: {r2_q2_other:.4f}\n'
    )
    print(txt)

    result = {
        'label': label, 'exp_config': args.exp_config, 'epoch': args.epoch,
        'eval_seed': args.eval_seed,  # P0-0: full RNG re-seed at main() top; this script feeds VE's ov_enc UNMASKED
                          # (matching the model's real forward(), which also reads it pre-mask --
                          # see model.py's SuperpositionNetworkProbeQValueEstimation.forward()), so
                          # there is no stochastic-masking randomness in this pipeline at all.
                          # Re-run twice on r4_move ep400 confirmed bit-identical R^2 regardless.
        'n_samples': int(q2_flat.shape[0]),
        'q2_to_self_pos_r2': r2_q2_self,
        'q2_to_other_pos_r2': r2_q2_other,
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, f'{label}.json'), 'w') as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(SAVE_DIR, f'{label}.txt'), 'w') as f:
        f.write(txt)
    print(f'Saved: {SAVE_DIR}/{label}.{{txt,json}}')


if __name__ == '__main__':
    main()
