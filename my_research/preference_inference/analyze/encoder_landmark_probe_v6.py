"""
Diagnostic requested by the user (2026-09-19) before committing to a FiLM
redesign: does the critic's CNN encoder actually represent all 4 landmark
colors, or only red/green (which was all v3/v4's 2-color reward ever
required)?

If the encoder's own features can't linearly predict blue/cyan pixel
fraction as well as red/green, then FiLM-modulating those features won't
help -- the encoder itself needs attention (capacity/training), not just
the conditioning mechanism downstream of it.

Method: sample many self-agent positions (self_random_other_stay env,
other fixed at center, matching the corner-scan / Q-map convention used
elsewhere in v6), extract the critic's CNNEncoder output (256-dim, the
same features that would feed into any FiLM modulation), and fit a
train/test Ridge regression from those features to each of the 4 true
landmark pixel fractions (computed directly from the same frame via
landmark_fractions -- ground truth, not the network's own guess).
Held-out R² per landmark is the metric: if red/green are high and
blue/cyan are much lower, the encoder has a real color-identity gap.

Usage (inside Docker):
    cd /work/my_research/preference_inference
    xvfb-run --auto-servernum python analyze/encoder_landmark_probe_v6.py \
        --critic_path data/model/v6_rl_critic_seed0_relabel.pth --label relabel
"""
import argparse
import json
import os
import sys

sys.path.insert(0, '/work')
sys.path.insert(0, '/work/simulation')

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import creator
from util import load_config
from my_research.preference_inference.model.rl_agent_sac_v6 import (
    CriticLSTM, landmark_fractions, LANDMARK_ORDER,
)

os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENV_CONFIG = '/work/simulation/config/collect/self_random_other_stay.yml'
SAVE_DIR = '/work/my_research/preference_inference/data/result/v6_baseline'
N_SAMPLES = 800
SEED = 0


def vision_to_tensor(v):
    v = v.astype(np.float32)
    return torch.tensor(np.transpose(v, (2, 0, 1))).unsqueeze(0).to(DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--critic_path', required=True)
    parser.add_argument('--label', default='critic')
    parser.add_argument('--n_samples', type=int, default=N_SAMPLES)
    args = parser.parse_args()

    rng = np.random.RandomState(SEED)

    env_cfg = load_config(ENV_CONFIG)
    env = creator.create_environment(env_cfg.environment)
    env.init()
    env.off_display()
    env.reset()
    env.world.set_camera('self')
    env.other_agent.p = np.array([0.0, 0.0])

    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(args.critic_path, map_location=DEVICE))
    critic.eval()

    features = []
    targets = []
    print(f'Sampling {args.n_samples} random self positions...')
    for i in range(args.n_samples):
        p = rng.uniform(-9.5, 9.5, size=2)
        env.self_agent.p = p
        env.world.draw(env.self_agent, env.other_agent)
        v = env.world.capture()
        frac = landmark_fractions(v)
        v_t = vision_to_tensor(v)
        with torch.no_grad():
            feat = critic.encoder(v_t)
        features.append(feat.squeeze(0).cpu().numpy())
        targets.append(frac)
        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{args.n_samples}')

    X = np.stack(features)
    Y = np.stack(targets)
    print(f'Feature matrix: {X.shape}, target matrix: {Y.shape}')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=SEED)

    results = {}
    print(f'\n=== encoder -> landmark-fraction Ridge regression, held-out R^2 ({args.label}) ===')
    for k, name in enumerate(LANDMARK_ORDER):
        model = Ridge(alpha=1.0)
        model.fit(X_train, Y_train[:, k])
        pred = model.predict(X_test)
        r2 = r2_score(Y_test[:, k], pred)
        results[name] = float(r2)
        print(f'  {name:6s}: R^2 = {r2:.4f}')

    out_path = os.path.join(SAVE_DIR, f'encoder_landmark_probe_{args.label}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
