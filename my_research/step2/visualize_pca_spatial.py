"""
visualize_pca_spatial.py

Replication of Noguchi et al. (2022) Fig. 4c:
  PCA of Shared Module hidden states, colored by agent spatial position.

Paper finding:
  h1 (Process-1) → organizes according to A-1's spatial location
  h2 (Process-2) → organizes according to A-2's spatial location
  Both show the same spatial structure → "shared spatial representation"

This script verifies that our Step-2 model (with Q input) also achieves
the same basic spatial representation, confirming the Superposition
mechanism is functioning before making claims about value learning.

Layout (3 panels):
  Left  : PCA of h1, colored by A-1's position
  Center: PCA of h2 (same axes), colored by A-2's position
  Right : 2D position-to-color legend

Color map (bilinear interpolation from 4 landmark corners):
  Red   (-9, +9)  top-left
  Cyan  (+9, +9)  top-right
  Green (-9, -9)  bottom-left
  Blue  (+9, -9)  bottom-right

Run:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_pca_spatial.py

Output:
    step2/pca_spatial_map.png
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

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
SAVE_PATH          = os.path.join(_HERE, 'pca_spatial_map.png')

N_EPISODES  = 50
MAX_STEPS   = 80
Q_SCALE     = 1.0

ARENA = 9.0  # arena half-size


def pos_to_color(x, y):
    """
    Bilinear interpolation of RGB from 4 landmark corner colors.
    Matches the color map in Noguchi et al. Fig. 4c.
      top-left  (-9, +9): Red   (1, 0, 0)
      top-right (+9, +9): Cyan  (0, 1, 1)
      bot-left  (-9, -9): Green (0, 0.8, 0)
      bot-right (+9, -9): Blue  (0, 0, 1)
    """
    u = np.clip((x + ARENA) / (2 * ARENA), 0, 1)   # 0=left,   1=right
    v = np.clip((y + ARENA) / (2 * ARENA), 0, 1)   # 0=bottom, 1=top

    c_bl = np.array([0.0, 0.7, 0.0])   # Green
    c_br = np.array([0.0, 0.0, 1.0])   # Blue
    c_tl = np.array([1.0, 0.0, 0.0])   # Red
    c_tr = np.array([0.0, 1.0, 1.0])   # Cyan

    return np.clip(
        (1-u)*(1-v)*c_bl + u*(1-v)*c_br +
        (1-u)*v   *c_tl + u*v   *c_tr,
        0, 1
    )


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
# Collect hidden states and agent positions
# =========================================================================
all_h1    = []
all_h2    = []
all_pos_a1 = []
all_pos_a2 = []

print(f"\nCollecting hidden states "
      f"({N_EPISODES} episodes x {MAX_STEPS} steps)...")

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

            x_in = {
                'self_vision':  v_t,
                'self_motion':  action,
                'other_motion': om_t,
            }
            _, h1, h2 = net(x_in, q_self, 0.0, 0.0)

        all_h1.append(h1.squeeze(0).cpu().numpy())
        all_h2.append(h2.squeeze(0).cpu().numpy())
        all_pos_a1.append(self_pos.copy())
        all_pos_a2.append(other_pos.copy())

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
        print(f"  {ep+1}/{N_EPISODES} done")

H1  = np.array(all_h1)
H2  = np.array(all_h2)
P_a1 = np.array(all_pos_a1)   # (T, 2)
P_a2 = np.array(all_pos_a2)   # (T, 2)

# Position → color for every timestep
C_a1 = np.array([pos_to_color(x, y) for x, y in P_a1])   # (T, 3)
C_a2 = np.array([pos_to_color(x, y) for x, y in P_a2])   # (T, 3)

print(f"\nCollected {len(H1)} steps.")

# =========================================================================
# PCA: fit on H1, project both onto same axes (following the paper)
# =========================================================================
pca = PCA(n_components=2)
pca.fit(H1)
H1_2d = pca.transform(H1)
H2_2d = pca.transform(H2)

var = pca.explained_variance_ratio_
print(f"PCA variance: PC1={var[0]:.3f}, PC2={var[1]:.3f}  "
      f"(total={var.sum():.3f})")

# =========================================================================
# Regression: linear prediction of A-1/A-2 positions from h1/h2
# (Analogous to Fig. 4d: MSE of linear regression over training)
# =========================================================================
print("\n=== Linear regression: hidden state → agent position (R²) ===")
print("  Paper finding: h1→A-1 pos and h2→A-2 pos both become accurate")
for name, H, pos_label, P in [
    ('h1 → A-1 position (own)',   H1, 'A-1', P_a1),
    ('h2 → A-2 position (own)',   H2, 'A-2', P_a2),
    ('h1 → A-2 position (cross)', H1, 'A-2', P_a2),
    ('h2 → A-1 position (cross)', H2, 'A-1', P_a1),
]:
    scores_x = cross_val_score(Ridge(), H, P[:, 0], cv=5, scoring='r2')
    scores_y = cross_val_score(Ridge(), H, P[:, 1], cv=5, scoring='r2')
    print(f"  {name}:  R²_x={scores_x.mean():.3f}  R²_y={scores_y.mean():.3f}")

# =========================================================================
# Figure: 3 panels (Process-1, Process-2, color legend)
# =========================================================================
fig = plt.figure(figsize=(16, 6))
fig.suptitle(
    'PCA of Shared Module hidden states — colored by spatial position\n'
    'Replication of Noguchi et al. (2022) Fig. 4c\n'
    f'PC1 var={var[0]:.2f}, PC2 var={var[1]:.2f}  '
    f'({N_EPISODES} episodes × {MAX_STEPS} steps)',
    fontsize=12
)

# ── Left: Process-1 (h1) ──────────────────────────────────────────────────
ax1 = fig.add_subplot(1, 3, 1)
ax1.scatter(H1_2d[:, 0], H1_2d[:, 1], c=C_a1, s=4, alpha=0.5)
ax1.set_title('Process-1  (h¹)\ncolored by A-1\'s position', fontsize=10)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.grid(True, alpha=0.2)

# ── Center: Process-2 (h2) ────────────────────────────────────────────────
ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(H2_2d[:, 0], H2_2d[:, 1], c=C_a2, s=4, alpha=0.5)
ax2.set_title('Process-2  (h²)\ncolored by A-2\'s position', fontsize=10)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.grid(True, alpha=0.2)

# ── Right: position-to-color legend ──────────────────────────────────────
ax3 = fig.add_subplot(1, 3, 3)
legend_size = 200
gx = np.linspace(-ARENA, ARENA, legend_size)
gy = np.linspace(-ARENA, ARENA, legend_size)
GX, GY = np.meshgrid(gx, gy)
legend_img = np.zeros((legend_size, legend_size, 3))
for i in range(legend_size):
    for j in range(legend_size):
        legend_img[i, j] = pos_to_color(GX[i, j], GY[i, j])
ax3.imshow(legend_img, extent=[-ARENA, ARENA, -ARENA, ARENA],
           origin='lower', aspect='equal')
# Landmark markers
for name, lx, ly, lc in [('Red', -9, 9, 'red'), ('Green', -9, -9, 'limegreen'),
                           ('Blue', 9, -9, 'blue'), ('Cyan', 9, 9, 'cyan')]:
    ax3.plot(lx, ly, '*', color=lc, markersize=14,
             markeredgecolor='black', markeredgewidth=0.6, label=name)
ax3.set_title('Position → Color legend\n(same as paper Fig. 4c)', fontsize=10)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.legend(fontsize=8, loc='center')

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {SAVE_PATH}")

# =========================================================================
# Check: are the color distributions similar between h1 and h2 PCA regions?
# =========================================================================
print("\n=== Visual check: do similar positions appear in similar PCA regions? ===")
print("  (Checking if corner-region points in PCA space have matching colors)")
for label, H_2d, C, P in [('Process-1', H1_2d, C_a1, P_a1),
                            ('Process-2', H2_2d, C_a2, P_a2)]:
    # Find points in each quadrant of PCA space and check avg position color
    q1 = (H_2d[:, 0] > np.median(H_2d[:, 0])) & (H_2d[:, 1] > np.median(H_2d[:, 1]))
    q3 = (H_2d[:, 0] < np.median(H_2d[:, 0])) & (H_2d[:, 1] < np.median(H_2d[:, 1]))
    if q1.sum() > 0 and q3.sum() > 0:
        c1 = C[q1].mean(axis=0)
        c3 = C[q3].mean(axis=0)
        print(f"  {label}: PCA quad1 avg color={c1.round(2)}, "
              f"quad3 avg color={c3.round(2)}")
        print(f"           color diff = {np.linalg.norm(c1-c3):.3f} "
              f"(higher = more spatial structure)")
