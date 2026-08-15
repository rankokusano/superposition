"""
Single-timestep (no-recurrence) probe-Q -> position regression (2026-08-15).

Follow-up to path_integration_check.py, after discovering that masking
p_mask_vision_self does NOT remove q1_vec's access to vision: R3-A's
compute_probe_q() reads the RAW, unmasked self_vision every timestep
(model.py forward(): q1_vec is computed from `sv` BEFORE
`util.mask(sv_enc, p_mask_vision_self)` is applied). This mirrors how
exp1_l1's m_t is also always given unmasked -- an intentional parallel,
not a bug -- but it means "mask vision, see if Q alone still works" can't
test what we actually want, because Q is inherently a function of the
CURRENT frame's state (not an incremental displacement like m_t), so
masking the separate sv_enc pathway never touches it.

The real question the user is asking: does h1->self=0.9086 reflect genuine
temporal path integration (the LSTM accumulating information over many
steps, the way it MUST for exp1_l1's displacement signal), or could the
same R^2 be achieved by a memoryless readout of the CURRENT timestep's Q
vector alone -- no recurrence, no history, just "what does this instant's
Q vector say"?

This script answers that directly: Ridge-regress self_position (and
other_position, for reference) on q1_vec ALONE, at matching (episode, t)
pairs, with NO LSTM/superposition_module/recurrence involved at all. If
this single-timestep R^2 is already close to h1->self's 0.9086, that is
strong evidence no integration is needed -- Q is close to already being a
position code, frame by frame. If it's much lower, the LSTM's recurrence
over q1_vec is doing real integrative work.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/probe_q_direct_regression.py --exp_config r3_a_direct --epoch 200
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
    parser.add_argument('--exp_config', default='r3_a_direct')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--n_episodes', type=int, default=3000)
    parser.add_argument('--label', default=None)
    args = parser.parse_args()
    label = args.label or f'{args.exp_config}_probe_q_direct_regression'

    exp_config = util.gen_exp_config(Args(args.exp_config, 0))
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(args.exp_config, 0), test=False)

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
    q_all = np.zeros((n_use, T, K), dtype=np.float32)

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
                q_vec = model.compute_probe_q(sv_raw)
                q_all[b0:b1, t] = q_vec.cpu().numpy()

    q_flat = q_all.reshape(n_use * T, K)
    sp_flat = sp_all.reshape(n_use * T, 2)
    op_flat = op_all.reshape(n_use * T, 2)

    r2_self = r2(q_flat, sp_flat)
    r2_other = r2(q_flat, op_flat)

    print(f'\n=== No-recurrence, single-timestep regression: q1_vec(t) -> position(t) ===')
    print(f'q1_vec -> self_position  R^2 = {r2_self:.4f}')
    print(f'q1_vec -> other_position R^2 = {r2_other:.4f}')
    print(f'\n(for comparison: h1 [after LSTM recurrence] -> self_position R^2 = 0.9086-0.9092 '
          f'per r3_a_direct_v2_r2.json / the probe-Q ablation baseline)')

    result = {
        'label': label,
        'exp_config': args.exp_config,
        'epoch': args.epoch,
        'n_episodes': n_use,
        'n_steps': T,
        'note': 'no LSTM/recurrence involved -- q1_vec(t) regressed directly onto '
                'position(t), same (episode,t) pairs, to test whether h1->self R^2 '
                'requires temporal integration or is already achievable from a single '
                'timestep\'s Q vector alone',
        'q_to_self_position_r2': r2_self,
        'q_to_other_position_r2': r2_other,
        'reference_h1_to_self_r2_with_recurrence': 0.9086,
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
