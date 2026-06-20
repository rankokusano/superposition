"""
visualize_pca_value.py

PCA visualization of h1/h2 hidden states colored by distance to goal
Directly analogous to Fig. 4c in Noguchi et al. (2022).

The paper showed: h1 encodes Agent-1's position, h2 encodes Agent-2's position
                  — both using the SAME representational structure.

This script shows: h1 encodes A-1's "value state" (distance to Red, its goal)
                   h2 encodes A-2's "value state" (distance to Green, its goal)
                   — if both show similar gradient structure, Process-2 has acquired
                     A-2's value representation analogous to how A-1 represents
                     its own value.

Purpose:
    Show that the Superposition Network with Q-input develops shared internal
    representations of value for both agents, just as the original network
    developed shared spatial representations.

What it shows:
    Left : PCA of h1 states, colored by A-1's distance to Red (-9,+9)
    Right: PCA of h2 states (projected onto same PCA axes), colored by
           A-2's distance to Green (-9,-9)
    If both show a clear gradient (far = one color, near = another color),
    the network has learned value-based representations for both agents.

Run:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_pca_value.py

Output:
    step2/pca_value_map.png
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '../../'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../'))
for p in [_HERE, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_green import GreenFollowerAgent
from model_q import SuperpositionNetworkWithQ

# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH    = os.path.join(_HERE, 'model_q.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')
SAVE_PATH          = os.path.join(_HERE, 'pca_value_map.png')

N_EPISODES  = 30    # episodes to collect hidden states
MAX_STEPS   = 80    # steps per episode

RED_POS   = np.array([-9.0,  9.0])
GREEN_POS = np.array([-9.0, -9.0])

Q_SCALE = 1.0

# =========================================================================
print(f"Device: {DEVICE}")

model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithQ(model_config).to(DEVICE)

if not os.path.exists(MODEL_SAVE_PATH):
    print("[WARNING] model_q.pth not found.")
    ckpt_iizuka = torch.load(
        os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth'),
        map_location=DEVICE)
    net.load_iizuka_weights(ckpt_iizuka, DEVICE)
else:
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt['net_state_dict'])
    Q_SCALE = ckpt.get('q_scale', 1.0)
    print(f"Loaded: episodes={ckpt.get('episodes','?')}, "
          f"final_loss={ckpt.get('final_loss', float('nan')):.5f}")

net.eval()

critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
critic.eval()

actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()

env_config = load_config(ENV_CONFIG_PATH)
env = creator.create_environment(env_config.environment)
env.init()
env.off_display()

green_follower = GreenFollowerAgent()

# =========================================================================
# Collect hidden states over many episodes
# =========================================================================
all_h1 = []   # (N, 128) — Process-1 hidden states
all_h2 = []   # (N, 128) — Process-2 hidden states
all_dist_red   = []  # A-1's distance to Red at each step
all_dist_green = []  # A-2's distance to Green at each step
all_pos_a1     = []  # A-1's (x,y) position
all_pos_a2     = []  # A-2's (x,y) position

print(f"\nCollecting hidden states ({N_EPISODES} episodes x {MAX_STEPS} steps)...")

for ep in range(N_EPISODES):
    env.reset()
    net.superposition_module.init_state(1)

    v_raw, _, _, sp_0, op_0 = env.step()
    self_pos  = sp_0.copy()
    other_pos = op_0.copy()
    actor_hidden  = None
    critic_hidden = None

    for step in range(MAX_STEPS):
        v_t = torch.FloatTensor(
            v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action, _, actor_hidden = actor.sample(v_t, actor_hidden)
            q_raw, critic_hidden = critic(v_t, action, critic_hidden)
            q_self = q_raw / Q_SCALE

            om_np = green_follower.get_action(other_pos)
            om_t  = torch.FloatTensor(om_np).unsqueeze(0).to(DEVICE)
            sm_t  = action

            x_in = {
                'self_vision':  v_t,
                'self_motion':  sm_t,
                'other_motion': om_t,
            }
            pred, h1, h2 = net(x_in, q_self, 0.0, 0.0)

        # Record
        all_h1.append(h1.squeeze(0).cpu().numpy())
        all_h2.append(h2.squeeze(0).cpu().numpy())
        all_dist_red.append(float(np.linalg.norm(self_pos - RED_POS)))
        all_dist_green.append(float(np.linalg.norm(other_pos - GREEN_POS)))
        all_pos_a1.append(self_pos.copy())
        all_pos_a2.append(other_pos.copy())

        # Update positions
        a_np = action.squeeze(0).cpu().numpy()
        new_self  = np.clip(self_pos  + a_np,  -9.5, 9.5)
        new_other = np.clip(other_pos + om_np, -9.5, 9.5)
        env.self_agent.p  = new_self
        env.other_agent.p = new_other
        v_next_raw, _, _, _, _ = env.step()
        net.superposition_module.detach_state()

        v_raw     = v_next_raw
        self_pos  = new_self
        other_pos = new_other

    if (ep + 1) % 10 == 0:
        print(f"  {ep+1}/{N_EPISODES} episodes done")

H1 = np.array(all_h1)           # (T, 128)
H2 = np.array(all_h2)           # (T, 128)
dist_red   = np.array(all_dist_red)    # (T,)
dist_green = np.array(all_dist_green)  # (T,)

print(f"\nCollected {len(H1)} total steps.")
print(f"  A-1 dist to Red   : min={dist_red.min():.2f}  max={dist_red.max():.2f}")
print(f"  A-2 dist to Green : min={dist_green.min():.2f}  max={dist_green.max():.2f}")

# =========================================================================
# PCA: fit on H1, then project both H1 and H2 onto same axes
# (Following the paper's approach: same PCA space for both processes)
# =========================================================================
pca = PCA(n_components=2)
pca.fit(H1)

H1_2d = pca.transform(H1)
H2_2d = pca.transform(H2)

var_explained = pca.explained_variance_ratio_
print(f"\nPCA variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}")

# =========================================================================
# Visualization
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    'PCA of Shared Module hidden states\n'
    'Analogy to Fig. 4c (Noguchi et al. 2022): position → value distance\n'
    f'PC1 var={var_explained[0]:.2f}, PC2 var={var_explained[1]:.2f}  '
    f'({N_EPISODES} episodes × {MAX_STEPS} steps)',
    fontsize=11
)

panels = [
    (axes[0], H1_2d, dist_red,
     'Process-1  (h¹)  —  A-1\'s hidden state\n'
     'Color = A-1\'s distance to Red (-9,+9)\n'
     'Gradient expected: far→near = color change'),
    (axes[1], H2_2d, dist_green,
     'Process-2  (h²)  —  A-2\'s hidden state\n'
     'Color = A-2\'s distance to Green (-9,-9)\n'
     'Similar gradient = value representation acquired'),
]

for ax, H_2d, dist, title in panels:
    sc = ax.scatter(
        H_2d[:, 0], H_2d[:, 1],
        c=dist, cmap='plasma_r',   # bright=near goal, dark=far
        s=4, alpha=0.5,
    )
    plt.colorbar(sc, ax=ax, label='Distance to goal  (bright = near)')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {SAVE_PATH}")

# =========================================================================
# Quantitative check: linear regression h -> distance to goal
# (Analogous to Fig. 4d regression analysis)
# =========================================================================
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

print("\n=== Linear regression: hidden state → distance to goal ===")
print("  (Lower R² is worse; higher R² = the hidden state encodes goal distance)")

for name, H, dist in [('h1 → dist_Red  ', H1, dist_red),
                       ('h2 → dist_Green', H2, dist_green),
                       ('h1 → dist_Green (cross)', H1, dist_green),
                       ('h2 → dist_Red   (cross)', H2, dist_red)]:
    reg = Ridge(alpha=1.0)
    scores = cross_val_score(reg, H, dist, cv=5, scoring='r2')
    print(f"  {name}: R² = {scores.mean():.4f} ± {scores.std():.4f}")
