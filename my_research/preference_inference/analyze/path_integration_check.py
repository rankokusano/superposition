"""
Direct path-integration check under complete vision masking (2026-08-15).

The paper's own training used p_mask_vision=0.99 (vision shown at t=0,
withheld ~99% of subsequent steps) specifically to FORCE path integration:
with vision almost never available, h1 can only track A-1's position by
integrating whatever process-1's input signal carries over time. For
exp1_l1 that input is m_t, a real displacement -- there's no way to track
position without literally integrating it. For R3-A(RL) that input is a
probe-Q vector, a function of *state* (self_vision), not of displacement --
so it's not obvious a priori that the same "integrate this signal to get
current position" mechanism is what's happening.

This script pushes masking to its logical extreme -- p_mask_vision=1.0 for
EVERY timestep including t=0 (the model never sees ANY real vision, not
even the paper's guaranteed first-frame anchor) -- and measures how much
h1->self Ridge R^2 survives. Run once per exp_config (r3_a_direct for
Q-driven, exp1_l1 for m_t-driven path integration) so the two numbers are
directly comparable: does Q substitute for m_t here, or does h1->self
collapse toward 0 once R3-A truly cannot see anything, unlike exp1_l1?

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/path_integration_check.py --exp_config r3_a_direct --epoch 200
    python analyze/path_integration_check.py --exp_config exp1_l1 --epoch 200 \
        --dataset_name self_random_other_stay
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
SAVE_DIR = 'data/result/baseline_v4'
T = 100
BATCH = 200

# default eval dataset per exp_config, matching how each model is normally tested
DEFAULT_DATASET = {
    'r3_a_direct': 'r2_a1random_a2rl',
    'exp1_l1': 'self_random_other_stay',
    'exp3': 'self_random_other_stay_periodic',
}


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
    parser.add_argument('--exp_config', required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--dataset_name', default=None)
    parser.add_argument('--n_episodes', type=int, default=3000)
    parser.add_argument('--label', default=None)
    parser.add_argument('--p_mask', type=float, default=1.0,
                         help='vision mask probability applied at EVERY timestep, '
                              'including t=0 (1.0 = vision never shown, ever)')
    args = parser.parse_args()
    dataset_name = args.dataset_name or DEFAULT_DATASET[args.exp_config]
    label = args.label or f'{args.exp_config}_path_integration_p{args.p_mask:.2f}'

    exp_config = util.gen_exp_config(Args(args.exp_config, 0))
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(args.exp_config, 0), test=False)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(DEVICE)
    util.load_model(model_dir, args.epoch, model)
    model.eval()

    data_h5 = f'/work/my_research/preference_inference/data/data/{dataset_name}/data.h5'
    with h5py.File(data_h5, 'r') as f:
        n_avail = f['train/self_vision'].shape[0]
        n_use = min(args.n_episodes, n_avail)
        print(f'exp_config={args.exp_config}  dataset={dataset_name}  '
              f'n_episodes={n_use}/{n_avail}  p_mask={args.p_mask}')
        sv_all = f['train/self_vision'][:n_use, :T]
        sp_all = f['train/self_position'][:n_use, :T]
        op_all = f['train/other_position'][:n_use, :T]
        has_motion = 'self_motion' in f['train']
        sm_all = f['train/self_motion'][:n_use, :T] if has_motion else None

    hidden_dim = model_config.superposition_module.hidden
    N = n_use
    h1_all = np.zeros((N, T, hidden_dim), dtype=np.float32)
    h2_all = np.zeros((N, T, hidden_dim), dtype=np.float32)

    torch.manual_seed(0)
    for b0 in range(0, N, BATCH):
        b1 = min(b0 + BATCH, N)
        bsz = b1 - b0
        model.init_state(bsz)
        for t in range(T):
            sv_np = sv_all[b0:b1, t].astype(np.float32)
            if sv_np.max() > 1.5:
                sv_np = sv_np / 255.0
            sv_scaled = util.scale_vision(sv_np)
            sv_t = torch.tensor(sv_scaled).permute(0, 3, 1, 2).float().to(DEVICE)

            x = {'self_vision': sv_t}
            if sm_all is not None:
                x['self_motion'] = torch.tensor(sm_all[b0:b1, t]).float().to(DEVICE)
            else:
                x['self_motion'] = torch.zeros(bsz, 2, device=DEVICE)

            with torch.no_grad():
                model(x, p_mask_vision_self=args.p_mask, p_mask_vision_other=args.p_mask)
                state = model.get_state()
                ss = state['self'][0]
                os_ = state['other'][0]

            h1_all[b0:b1, t] = ss.detach().cpu().numpy()
            h2_all[b0:b1, t] = os_.detach().cpu().numpy()

    D = hidden_dim
    h1_flat = h1_all.reshape(N * T, D)
    h2_flat = h2_all.reshape(N * T, D)
    sp_flat = sp_all.reshape(N * T, 2)
    op_flat = op_all.reshape(N * T, 2)

    result = {
        'label': label,
        'exp_config': args.exp_config,
        'dataset_name': dataset_name,
        'epoch': args.epoch,
        'p_mask': args.p_mask,
        'n_episodes': N,
        'n_steps': T,
        'h1_to_self': r2(h1_flat, sp_flat),
        'h1_to_other': r2(h1_flat, op_flat),
        'h2_to_self': r2(h2_flat, sp_flat),
        'h2_to_other': r2(h2_flat, op_flat),
    }
    result.update(util.gen_result_metadata(
        exp_config_name=args.exp_config, seed=0, dataset_name=dataset_name))

    print(f"h1->self={result['h1_to_self']:.4f}  h1->other={result['h1_to_other']:.4f}  "
          f"h2->self={result['h2_to_self']:.4f}  h2->other={result['h2_to_other']:.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{label}.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
