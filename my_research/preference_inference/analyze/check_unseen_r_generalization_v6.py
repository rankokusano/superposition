"""
S5 pre-flight check (2026-10-02, user request / docs/v6_experiment_log.md
§26.5, §27.4 "unseen r visual caveat -- re-check before S5"): does the
critic actually used by S2-S4 (seed2, FiLM, norelabel, curriculum) respond
to UNSEEN, continuous r values by genuinely identifying landmark colors,
or does its "unseen" generalisation (previously only checked at a single
point, r=(0,+1,-1,0)) just ride a coordinate-axis gradient that happens to
agree with that one r's preferred/disliked landmark pair?

The previous unseen check is confounded: Green=(-9,-9) and Blue=(9,-9)
differ ONLY in x, so "prefers Green, dislikes Blue" is indistinguishable
from "a plain left-bright/right-dark gradient" using that r alone (already
flagged in SS26.5/SS22 -- plane-fit R^2 couldn't separate the two
explanations either, since a 3-parameter plane fit to a handful of corner
points is never very diagnostic).

This script picks r vectors designed to break that confound:
  - diag_RB  = (1,0,-1,0): prefers Red(-9,9), dislikes Blue(9,-9) --
    a genuine diagonal pair (differ in BOTH x and y)
  - diag_GC  = (0,1,0,-1): prefers Green(-9,-9), dislikes Cyan(9,9) --
    the other diagonal
  - checkerboard = (0.5,-0.5,0.5,-0.5): prefers Red+Blue (one diagonal's
    two opposite corners), dislikes Green+Cyan (the other diagonal's two
    opposite corners). No LINEAR function of (x,y) can produce "high at
    both ends of one diagonal, low at both ends of the other" -- that
    needs an xy interaction term. If the critic's spatial map shows high
    Q concentrated at Red AND Blue specifically (not a smooth gradient),
    that is strong, hard-to-fake evidence of genuine per-landmark
    evaluation rather than a coordinate-gradient artifact.

Reuses plot_q_map_v6.py's grid-computation code directly (not duplicated).

Usage (inside Docker, needs the simulation renderer -- xvfb):
    cd /work/my_research/preference_inference
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python analyze/check_unseen_r_generalization_v6.py --seed 2 --film --tag _film_norelabel_curriculum
"""
import argparse
import json
import os
import sys

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import creator
from util import load_config
from my_research.preference_inference.model.rl_agent_sac_v6 import ActorLSTM, CriticLSTM
from plot_q_map_v6 import (
    compute_q_grid, plot_q_map, r_to_tensor, LANDMARKS, ENV_CONFIG, DEVICE, MODEL_DIR,
)

SAVE_DIR = '/work/my_research/preference_inference/data/result/v6_baseline'

EXTRA_R_CONDITIONS = [
    ('diag_RB (1,0,-1,0) Red-vs-Blue', np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32), 'Red', 'Blue'),
    ('diag_GC (0,1,0,-1) Green-vs-Cyan', np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32), 'Green', 'Cyan'),
    ('checkerboard (.5,-.5,.5,-.5) RedBlue-vs-GreenCyan',
     np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32), ('Red', 'Blue'), ('Green', 'Cyan')),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--tag', default='_film_norelabel_curriculum')
    ap.add_argument('--film', action='store_true', default=True)
    args = ap.parse_args()

    config = load_config(ENV_CONFIG)
    env = creator.create_environment(config.environment)
    env.init()
    env.off_display()

    actor = ActorLSTM(film=args.film).to(DEVICE)
    actor.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f'v6_rl_actor_seed{args.seed}{args.tag}.pth'), map_location=DEVICE))
    actor.eval()
    critic = CriticLSTM(film=args.film).to(DEVICE)
    critic.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f'v6_rl_critic_seed{args.seed}{args.tag}.pth'), map_location=DEVICE))
    critic.eval()

    grids = {}
    stats = {}
    for name, r_vec, pref, dislike in EXTRA_R_CONDITIONS:
        print(f'Computing Q-map for r = {name} ...')
        r_t = r_to_tensor(r_vec)
        q_grid = compute_q_grid(env, actor, critic, r_t)
        grids[name] = q_grid
        stats[name] = dict(min=float(q_grid.min()), max=float(q_grid.max()), std=float(q_grid.std()))
        # corner readout at the 4 landmark positions (nearest grid point)
        corner_vals = {}
        for lname, lpos in LANDMARKS.items():
            xi = int(np.argmin(np.abs(np.linspace(-9, 9, q_grid.shape[0]) - lpos[0])))
            yi = int(np.argmin(np.abs(np.linspace(-9, 9, q_grid.shape[1]) - lpos[1])))
            corner_vals[lname] = float(q_grid[xi, yi])
        stats[name]['corners'] = corner_vals
        print(f'  corners: ' + '  '.join(f'{k}={v:.3f}' for k, v in corner_vals.items()))

    fig, axes = plt.subplots(1, len(EXTRA_R_CONDITIONS), figsize=(6 * len(EXTRA_R_CONDITIONS), 5))
    for ax, (name, r_vec, pref, dislike) in zip(axes, EXTRA_R_CONDITIONS):
        s = stats[name]
        im = plot_q_map(grids[name], f'r = {name}\nstd={s["std"]:.4f}', ax, s['min'], s['max'])
        plt.colorbar(im, ax=ax, label='Q-value')
    plt.suptitle(f'S5 pre-flight: unseen continuous-r generalisation, confound-broken '
                 f'(seed{args.seed}{args.tag})', fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, f'unseen_r_generalization_seed{args.seed}{args.tag}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved: {out_path}')

    stats_path = os.path.join(SAVE_DIR, f'unseen_r_generalization_seed{args.seed}{args.tag}_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'Saved: {stats_path}')


if __name__ == '__main__':
    main()
