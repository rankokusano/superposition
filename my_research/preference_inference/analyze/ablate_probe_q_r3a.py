"""
R3-A(RL) probe-Q inference-time ablations (2026-08-15).

Motivating concern (user, 2026-08-15): exp3's process-1 input was m_t, A-1's
own real displacement -- the LSTM had no choice but to use it for path
integration, so h1->self being high was necessarily a path-integration
result. R3-A's process-1 input is instead a K=8 probe-Q vector, a function
of *state* (self_vision), not of displacement. It's possible for h1->self
to come out high (0.9086) without any path integration happening at all --
e.g. if the Shared Module is just using the Q vector as an opaque
state-identifier (which state produces this Q pattern) rather than
integrating directional/value structure over time. The near-identical
result under full weight_ih random-init vs. partial exp1_l1 transfer
(0.9086 vs 0.9095) is consistent with this: if warm-starting from
exp1_l1's motion-conditioned SM weights doesn't matter, R3-A's SM may be
learning an entirely different (non-path-integration) mechanism regardless
of initialization.

Four INFERENCE-ONLY interventions on the probe-Q vector fed into R3-A(RL)'s
already-trained (frozen) Shared Module -- no additional training anywhere:

  1. rotate90  -- torch.roll the already-computed 8-dim vector by K/4=2
     positions (each probe is 45 degrees apart, so a 2-position cyclic
     shift is a 90-degree rotation of the probe circle). A structured,
     norm-preserving relabeling of which index means which direction.
  2. shuffle   -- apply ONE FIXED random permutation (seeded, same
     permutation used for every episode/timestep of this run) to the 8
     elements. Like rotate90, this exactly preserves the vector's L2 norm
     and full information content, differing only in how unstructured the
     index relabeling is (cyclic vs. arbitrary).
  3. critic_swap -- evaluate the SAME 8 probe actions against the SAME
     A-1 self_vision trajectory, but through A-2's frozen critic
     (v3_rl_a2_critic.pth) instead of A-1's own (v3_rl_critic.pth), then
     normalize with A-1's own (mu, sigma) exactly as usual. Isolates
     "does h1 need A-1's OWN value structure specifically" from "does h1
     just need SOME structured critic-shaped 8-dim signal."
  4. constant  -- collapse the 8-dim vector to a single scalar (the true
     vector's own per-timestep mean) broadcast to all 8 positions. Removes
     ALL directional/K-dim structure while preserving the state-varying
     overall Q *level* (so this isolates "directional structure" from
     "overall magnitude," rather than removing both at once).

For each condition (plus an unmodified baseline), measures:
  - self_vision L1 reconstruction error (mean over all unmasked-input
    timesteps is meaningless since masking is stochastic every timestep
    regardless of condition -- reported over ALL timesteps, matching how
    the model is actually trained/evaluated)
  - the same 4-axis Ridge R^2 (h1->self, h1->other, h2->self, h2->other)
    as analyze/regression_baseline_v4.py, computed here directly from a
    manually-replicated forward pass (since we need to inject a modified
    probe-Q vector mid-forward-pass, which the model's own .forward()
    doesn't expose a hook for).

Judgment criterion (per user): if h1->self barely drops under conditions
1-3, h1 is not using the probe-Q vector's directional/value structure --
it's using it as an opaque state identifier, which would undercut R4's
premise that VE's output needs to function as a genuine value signal.

Usage (inside Docker, from /work/my_research/preference_inference):
    python analyze/ablate_probe_q_r3a.py --exp_config r3_a_direct --epoch 200
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
    sys.path.insert(0, _PI_DIR)  # must come after '/work' so this package's
                                 # `model`/`util` shadow the root-level ones

import model as models  # noqa
from model import util as model_util  # noqa
import util  # noqa
from my_research.rl_agent_sac import CriticLSTM  # noqa

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_H5 = '/work/my_research/preference_inference/data/data/r2_a1random_a2rl/data.h5'
A2_CRITIC_PATH = '/work/my_research/preference_inference/data/model/v3_rl_a2_critic.pth'
SAVE_DIR = 'data/result/baseline_v4'
N_EPISODES = 3000  # matches the full pool used by regression_baseline_v4.py's N
T = 100
BATCH = 200
MASK_SEED = 20260815  # fixed so every condition sees the identical masking realization
SHUFFLE_SEED = 20260815

CONDITIONS = ['baseline', 'rotate90', 'shuffle', 'critic_swap', 'constant']


class Args:
    def __init__(self, exp_config, seed):
        self.exp_config = exp_config
        self.seed = seed


def r2(X, Y):
    reg = Ridge(alpha=1.0)
    reg.fit(X, Y)
    pred = reg.predict(X)
    return r2_score(Y, pred)


def apply_condition(q1_vec, condition, sv_raw, model, ctx):
    k = q1_vec.size(1)
    if condition == 'baseline':
        return q1_vec
    elif condition == 'rotate90':
        return torch.roll(q1_vec, shifts=k // 4, dims=1)
    elif condition == 'shuffle':
        return q1_vec[:, ctx['shuffle_perm']]
    elif condition == 'critic_swap':
        b = sv_raw.size(0)
        v_rep = sv_raw.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(
            b * k, *sv_raw.shape[1:])
        a_rep = model.probe_actions.unsqueeze(0).expand(b, -1, -1).reshape(b * k, 2)
        with torch.no_grad():
            q, _ = ctx['a2_critic'](v_rep, a_rep, hidden=None)
        q = q.reshape(b, k)
        return torch.tanh((q - model.q_mu) / model.q_sigma)
    elif condition == 'constant':
        m = q1_vec.mean(dim=1, keepdim=True)
        return m.expand_as(q1_vec)
    else:
        raise ValueError(condition)


def run_condition(model, sv_all, sp_all, op_all, condition, ctx, p_mask_vision, hidden_dim):
    N = sv_all.shape[0]
    h1_all = np.zeros((N, T, hidden_dim), dtype=np.float32)
    h2_all = np.zeros((N, T, hidden_dim), dtype=np.float32)
    vision_l1_sum = 0.0
    vision_l1_n = 0

    torch.manual_seed(MASK_SEED)  # identical masking draws across all conditions

    for b0 in range(0, N, BATCH):
        b1 = min(b0 + BATCH, N)
        bsz = b1 - b0
        model.init_state(bsz)
        for t in range(T):
            sv_np = sv_all[b0:b1, t].astype(np.float32)
            if sv_np.max() > 1.5:  # uint8 source
                sv_np = sv_np / 255.0
            sv_scaled = util.scale_vision(sv_np)
            sv_t = torch.tensor(sv_scaled).permute(0, 3, 1, 2).float().to(DEVICE)

            p_mask = 0.0 if t == 0 else p_mask_vision

            with torch.no_grad():
                sv_enc = model.self_vision_encoder_module(sv_t)
                ov_enc = model.other_vision_encoder_module(sv_t)
                sv_raw = (sv_t + 1) / 2
                q1_vec_true = model.compute_probe_q(sv_raw)
                q1_vec = apply_condition(q1_vec_true, condition, sv_raw, model, ctx)
                q2_vec = torch.zeros_like(q1_vec)

                sv_enc_m = model_util.mask(sv_enc, p_mask)
                ov_enc_m = model_util.mask(ov_enc, p_mask)

                ss, os_ = model.superposition_module(sv_enc_m, q1_vec, ov_enc_m, q2_vec)
                so = model.integration_module(ss, os_)  # eval mode: dropout is a no-op
                vision_pred = model.vision_decoder_module(so)

            vision_l1_sum += (vision_pred - sv_t).abs().mean().item() * bsz
            vision_l1_n += bsz

            h1_all[b0:b1, t] = ss.detach().cpu().numpy()
            h2_all[b0:b1, t] = os_.detach().cpu().numpy()

    D = hidden_dim
    h1_flat = h1_all.reshape(N * T, D)
    h2_flat = h2_all.reshape(N * T, D)
    sp_flat = sp_all[:, :T].reshape(N * T, 2)
    op_flat = op_all[:, :T].reshape(N * T, 2)

    result = {
        'condition': condition,
        'vision_l1_error': vision_l1_sum / vision_l1_n,
        'h1_to_self': r2(h1_flat, sp_flat),
        'h1_to_other': r2(h1_flat, op_flat),
        'h2_to_self': r2(h2_flat, sp_flat),
        'h2_to_other': r2(h2_flat, op_flat),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_config', default='r3_a_direct')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES)
    parser.add_argument('--label', default=None)
    args = parser.parse_args()
    label = args.label or f'{args.exp_config}_probe_q_ablation'

    exp_config = util.gen_exp_config(Args(args.exp_config, 0))
    model_config = util.gen_model_config(exp_config)
    result_dir, model_dir, log_dir = util.gen_dirs(Args(args.exp_config, 0), test=False)

    model = getattr(models, exp_config.model.name)(model_config)
    model.to(DEVICE)
    util.load_model(model_dir, args.epoch, model)
    model.eval()

    hidden_dim = model_config.superposition_module.hidden
    p_mask_vision = exp_config.p_mask_vision if 'p_mask_vision' in exp_config else 0.99

    a2_critic = CriticLSTM()
    a2_critic.load_state_dict(torch.load(A2_CRITIC_PATH, map_location=DEVICE))
    a2_critic.to(DEVICE)
    a2_critic.eval()
    for p in a2_critic.parameters():
        p.requires_grad = False

    k = model.probe_actions.size(0)
    shuffle_rng = np.random.RandomState(SHUFFLE_SEED)
    shuffle_perm = torch.tensor(shuffle_rng.permutation(k), dtype=torch.long, device=DEVICE)
    ctx = {'a2_critic': a2_critic, 'shuffle_perm': shuffle_perm}

    with h5py.File(DATA_H5, 'r') as f:
        n_avail = f['train/self_vision'].shape[0]
        n_use = min(args.n_episodes, n_avail)
        print(f'Loading {n_use}/{n_avail} episodes from {DATA_H5} ...')
        sv_all = f['train/self_vision'][:n_use, :T]
        sp_all = f['train/self_position'][:n_use, :T]
        op_all = f['train/other_position'][:n_use, :T]

    results = []
    for condition in CONDITIONS:
        print(f'--- condition: {condition} ---')
        res = run_condition(model, sv_all, sp_all, op_all, condition, ctx,
                             p_mask_vision, hidden_dim)
        print(f"  vision_l1={res['vision_l1_error']:.4f}  "
              f"h1->self={res['h1_to_self']:.4f}  h1->other={res['h1_to_other']:.4f}  "
              f"h2->self={res['h2_to_self']:.4f}  h2->other={res['h2_to_other']:.4f}")
        results.append(res)

    baseline_h1_self = results[0]['h1_to_self']
    print(f"\n=== Judgment: does h1->self drop under conditions 1-3 "
          f"(rotate90/shuffle/critic_swap)? ===")
    print(f'baseline h1->self = {baseline_h1_self:.4f}')
    for res in results[1:4]:
        drop = baseline_h1_self - res['h1_to_self']
        drop_pct = 100 * drop / baseline_h1_self if baseline_h1_self else float('nan')
        print(f"  {res['condition']:12s}: h1->self={res['h1_to_self']:.4f}  "
              f"drop={drop:+.4f} ({drop_pct:+.1f}%)")
    constant_res = results[4]
    drop = baseline_h1_self - constant_res['h1_to_self']
    print(f"  constant    : h1->self={constant_res['h1_to_self']:.4f}  "
          f"drop={drop:+.4f} ({100*drop/baseline_h1_self:+.1f}%)  "
          f"[collapses directional structure but keeps state-varying overall level]")

    output = {
        'label': label,
        'exp_config': args.exp_config,
        'epoch': args.epoch,
        'n_episodes': n_use,
        'n_steps': T,
        'mask_seed': MASK_SEED,
        'shuffle_seed': SHUFFLE_SEED,
        'shuffle_permutation': shuffle_perm.cpu().tolist(),
        'a2_critic_path': A2_CRITIC_PATH,
        'results': results,
    }
    output.update(util.gen_result_metadata(
        exp_config_name=args.exp_config, seed=0, dataset_name='r2_a1random_a2rl'))

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f'{label}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
