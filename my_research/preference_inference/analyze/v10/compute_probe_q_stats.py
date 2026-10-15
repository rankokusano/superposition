"""
v10 B3: probe-Q normalisation statistics (mu, sigma) = mean and (population)
std of the raw critic Q(s, a_probe) over all frames x the 8 probe actions of
A-1's own vision (`self_vision`) in a dataset split -- the v4 procedure
(v4_experiment_log Sec.4.2: "A-1 の実データのみから"; known_confounds.md
2026-08-14: "from self_vision + 8-probe critic evaluation"). The exact v4
script is not in the repository, so this re-implementation is first checked
against v4's values (v3 critic on r2_a1random_a2rl: mu 1.9233, sigma 0.9466).
Critic evaluated per frame with hidden=None, as in the model (compute_probe_q).
"""

import argparse
import json
import math
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, '/work')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v10 import OUT_ROOT, PI_DIR, git_hash  # noqa
from my_research.rl_agent_sac import CriticLSTM  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--critic', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--label', required=True)
    args = ap.parse_args()
    dev = torch.device(args.device)
    critic = CriticLSTM().to(dev)
    critic.load_state_dict(torch.load(os.path.join(PI_DIR, args.critic), map_location=dev))
    critic.eval()
    probes = torch.tensor([[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)],
                          dtype=torch.float32, device=dev)
    s1 = s2 = 0.0
    n = 0
    with h5py.File(os.path.join(PI_DIR, 'data', 'data', args.dataset, 'data.h5'), 'r') as f:
        sv = f[f'{args.split}/self_vision']
        N, T = sv.shape[:2]
        for i in range(0, N, 50):
            v = sv[i:i + 50].reshape(-1, *sv.shape[2:])
            x = torch.from_numpy(v.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev)
            b = x.shape[0]
            with torch.no_grad():
                q, _ = critic(x.unsqueeze(1).expand(-1, 8, -1, -1, -1).reshape(b * 8, *x.shape[1:]),
                              probes.repeat(b, 1), hidden=None)
            q = q.double().cpu().numpy().ravel()
            s1 += q.sum()
            s2 += (q * q).sum()
            n += q.size
    mu = s1 / n
    sigma = math.sqrt(s2 / n - mu * mu)
    out = {'label': args.label, 'critic': args.critic, 'dataset': args.dataset, 'split': args.split,
           'n_frames': int(N * T), 'n_values': int(n), 'mu': mu, 'sigma': sigma, 'git_commit': git_hash(),
           'method': 'mean / population std of raw Q over all frames x 8 probes, hidden=None'}
    path = os.path.join(OUT_ROOT, 'b3', f'probe_q_stats_{args.label}.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
