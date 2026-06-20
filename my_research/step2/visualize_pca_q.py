"""
visualize_pca_q.py

PCA of h1/h2 hidden states colored by Q values (not landmark distances).

Left : h1 (Process-1) colored by Critic Q  — A-1's actual value estimate
Right: h2 (Process-2) colored by ValueEstimator Q — A-2's inferred value

If both panels show similar gradient structure (low Q in one region,
high Q in another), it means the network has developed a shared internal
representation of value for both agents.

Axes:
  PC1, PC2 = principal components of h1 (fitted on h1, applied to both)
  Color     = Q value at each timestep (not a positional proxy)

Analogy to Noguchi et al. Fig.4c, but for VALUE rather than POSITION.

Run:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_pca_q.py

Output:
    step2/pca_q_map.png
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
SAVE_PATH          = os.path.join(_HERE, 'pca_q_map.png')

N_EPISODES  = 30
MAX_STEPS   = 80
Q_SCALE     = 1.0

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
# Collect h1, h2, and Q values over many episodes
# =========================================================================
all_h1     = []   # Process-1 hidden states (128-dim)
all_h2     = []   # Process-2 hidden states (128-dim)
all_q_critic = []    # Critic Q for A-1 at each step
all_q_ve     = []    # ValueEstimator Q for A-2 at each step

print(f"\nCollecting hidden states + Q values "
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
            # A-1 action and Critic Q
            action, _, actor_hidden = actor.sample(v_t, actor_hidden)
            q_critic, critic_hidden = critic(v_t, action, critic_hidden)

            q_self = q_critic / Q_SCALE

            # A-2 motion (GreenFollower)
            om_np = green_follower.get_action(other_pos)
            om_t  = torch.FloatTensor(om_np).unsqueeze(0).to(DEVICE)
            sm_t  = action

            # Forward pass through Superposition Network
            x_in = {
                'self_vision':  v_t,
                'self_motion':  sm_t,
                'other_motion': om_t,
            }
            pred, h1, h2 = net(x_in, q_self, 0.0, 0.0)

            # ValueEstimator Q for A-2
            q_ve = net.value_estimator(h2, om_t)

        # Record
        all_h1.append(h1.squeeze(0).cpu().numpy())
        all_h2.append(h2.squeeze(0).cpu().numpy())
        all_q_critic.append(float(q_critic.item()))
        all_q_ve.append(float(q_ve.item()))

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

H1 = np.array(all_h1)
H2 = np.array(all_h2)
Q_critic = np.array(all_q_critic)
Q_ve     = np.array(all_q_ve)

print(f"\nCollected {len(H1)} steps.")
print(f"  Critic Q   : mean={Q_critic.mean():.3f}  "
      f"min={Q_critic.min():.3f}  max={Q_critic.max():.3f}")
print(f"  ValueEst Q : mean={Q_ve.mean():.3f}  "
      f"min={Q_ve.min():.3f}  max={Q_ve.max():.3f}")

# =========================================================================
# PCA: fit on H1, project both H1 and H2 onto same axes
# =========================================================================
pca = PCA(n_components=2)
pca.fit(H1)
H1_2d = pca.transform(H1)
H2_2d = pca.transform(H2)

var = pca.explained_variance_ratio_
print(f"\nPCA (fitted on h1): PC1={var[0]:.3f}, PC2={var[1]:.3f}")

# =========================================================================
# Visualization
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(
    'PCA of Shared Module hidden states — colored by Q value\n'
    'Each point = 1 timestep.  '
    f'PC1={var[0]:.2f}, PC2={var[1]:.2f}  '
    f'({N_EPISODES} ep x {MAX_STEPS} steps)',
    fontsize=12
)

panels = [
    (axes[0], H1_2d, Q_critic,
     'Process-1  h¹  —  colored by Critic Q\n'
     'A-1\'s value as estimated by its own Critic\n'
     '(trained in Step 1 via SAC)'),
    (axes[1], H2_2d, Q_ve,
     'Process-2  h²  —  colored by ValueEstimator Q\n'
     'A-2\'s value as inferred by Process-2\n'
     '(learned in Step 2 via prediction loss)'),
]

for ax, H_2d, Q, title in panels:
    # Use percentile clipping to avoid outlier color distortion
    vmin, vmax = np.percentile(Q, 5), np.percentile(Q, 95)
    sc = ax.scatter(
        H_2d[:, 0], H_2d[:, 1],
        c=Q,
        cmap='plasma',
        vmin=vmin, vmax=vmax,
        s=5, alpha=0.5,
    )
    cb = plt.colorbar(sc, ax=ax, label='Q value')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {SAVE_PATH}")

# =========================================================================
# Correlation: does Q value correlate with h1/h2 PC components?
# =========================================================================
print("\n=== Correlation between PCA components and Q values ===")
print("  (High |r| means Q value is linearly organized along that PC axis)")
for pc_idx in [0, 1]:
    r1 = float(np.corrcoef(H1_2d[:, pc_idx], Q_critic)[0, 1])
    r2 = float(np.corrcoef(H2_2d[:, pc_idx], Q_ve)[0, 1])
    print(f"  PC{pc_idx+1}:  h1-Critic r={r1:+.3f}   h2-ValueEst r={r2:+.3f}")

# Linear regression: how well can each hidden state predict its own Q?
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

print("\n=== Linear probe: hidden state → Q value (R²) ===")
print("  Key question: is the Q value linearly decodable from h?")
for name, H, Q in [
    ('h1 -> Critic Q    (own)',   H1, Q_critic),
    ('h2 -> ValueEst Q  (own)',   H2, Q_ve),
    ('h1 -> ValueEst Q  (cross)', H1, Q_ve),
    ('h2 -> Critic Q    (cross)', H2, Q_critic),
]:
    scores = cross_val_score(Ridge(alpha=1.0), H, Q, cv=5, scoring='r2')
    print(f"  {name}: R² = {scores.mean():.4f} ± {scores.std():.4f}")
