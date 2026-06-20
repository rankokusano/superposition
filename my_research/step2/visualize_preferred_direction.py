"""
visualize_preferred_direction.py

Preferred Action Direction Map (goal-agnostic)

At each grid position, finds the action direction that maximizes Q
WITHOUT specifying what the goal should be.

The network reveals its own preference.
If learning succeeded:
  - A-1 Critic arrows converge toward Red (-9, +9)
  - A-2 ValueEstimator arrows converge toward Green (-9, -9)

Left panel : A-1 Critic
Right panel: A-2 ValueEstimator

Arrow color/length: preference strength = Q_max - Q_min among 12 directions
Background: heatmap of preference strength (how confident the preference is)

Run:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_preferred_direction.py

Output:
    step2/preferred_direction_map.png
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
import matplotlib.cm as cm

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from rl_agent_sac import ActorLSTM, CriticLSTM
from model_q import SuperpositionNetworkWithQ

# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH    = os.path.join(_HERE, 'model_q.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')
SAVE_PATH          = os.path.join(_HERE, 'preferred_direction_map.png')

GRID_SIZE    = 15     # 15x15 = 225 positions
N_ANGLES     = 12     # candidate directions at each position
WARMUP_STEPS = 10     # steps to warm up h² for A-2

ACTION_SCALE_A1 = 0.7   # action magnitude for Critic (within tanh range)
ACTION_SCALE_A2 = 1.0   # action magnitude for ValueEstimator (GreenFollower speed)

# Fixed positions for the "other" agent in each panel
A2_FIXED_POS = np.array([-9.0, -9.0])  # A-2 at green landmark for A-1 panel
A1_FIXED_POS = np.array([ 0.0,  0.0])  # A-1 at center for A-2 panel

LANDMARKS = [
    ('Red',   -9,  9, 'red'),
    ('Green', -9, -9, 'limegreen'),
    ('Blue',   9, -9, 'blue'),
    ('Cyan',   9,  9, 'cyan'),
]

GREEN_POS = np.array([-9.0, -9.0])

# Candidate action directions: 12 unit vectors at 30° intervals
ANGLES = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
CAND_DIRS = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1)  # (N_ANGLES, 2)

Q_SCALE = 1.0


def warmup_action(pos):
    """Unit vector toward green (mimics GreenFollower, for h² warmup only)."""
    delta = GREEN_POS - pos
    dist = np.linalg.norm(delta)
    if dist < 1e-6:
        return np.array([0.0, 1.0])
    return delta / dist


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
          f"final_loss={ckpt.get('final_loss', float('nan')):.5f}, "
          f"Q_SCALE={Q_SCALE}")

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

# =========================================================================
xs = np.linspace(-9, 9, GRID_SIZE)
ys = np.linspace(-9, 9, GRID_SIZE)

# Preferred direction (unit vector) at each grid position
pref_U_a1 = np.zeros((GRID_SIZE, GRID_SIZE))  # x-component
pref_V_a1 = np.zeros((GRID_SIZE, GRID_SIZE))  # y-component
pref_S_a1 = np.zeros((GRID_SIZE, GRID_SIZE))  # preference strength

pref_U_a2 = np.zeros((GRID_SIZE, GRID_SIZE))
pref_V_a2 = np.zeros((GRID_SIZE, GRID_SIZE))
pref_S_a2 = np.zeros((GRID_SIZE, GRID_SIZE))

print(f"\nComputing preferred directions "
      f"({GRID_SIZE}x{GRID_SIZE} grid, {N_ANGLES} angles, "
      f"warmup={WARMUP_STEPS})...")

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        pos = np.array([x, y])

        with torch.no_grad():

            # ── A-1 Critic: find preferred direction ─────────────────────
            v_raw = env.capture_at_pos(pos, A2_FIXED_POS)
            v_t = torch.FloatTensor(
                v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            q_vals_a1 = np.zeros(N_ANGLES)
            for k, d in enumerate(CAND_DIRS):
                a = torch.FloatTensor(
                    d * ACTION_SCALE_A1).unsqueeze(0).to(DEVICE)
                q_vals_a1[k] = critic(v_t, a)[0].item()

            best_k = int(np.argmax(q_vals_a1))
            pref_U_a1[j, i] = CAND_DIRS[best_k, 0]
            pref_V_a1[j, i] = CAND_DIRS[best_k, 1]
            pref_S_a1[j, i] = q_vals_a1.max() - q_vals_a1.min()

            # ── A-2 ValueEstimator: warm up h², then find preferred dir ──
            v_raw2 = env.capture_at_pos(A1_FIXED_POS, pos)
            v_t2 = torch.FloatTensor(
                v_raw2.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            # Warmup: simulate A-2 at this position moving toward green
            om_warmup = torch.FloatTensor(
                warmup_action(pos) * ACTION_SCALE_A2
            ).unsqueeze(0).to(DEVICE)

            net.superposition_module.init_state(1)
            actor_h = None
            for _ in range(WARMUP_STEPS):
                a_tmp, _, actor_h = actor.sample(v_t2, actor_h)
                q_s, _ = critic(v_t2, a_tmp)
                x_in = {
                    'self_vision':  v_t2,
                    'self_motion':  a_tmp,
                    'other_motion': om_warmup,
                }
                _, _, h2 = net(x_in, q_s / Q_SCALE, 0.0, 0.0)
                net.superposition_module.detach_state()

            h2_w = net.superposition_module.state['other'].hidden

            # Sample all 12 directions and find which maximizes Q_VE
            q_vals_a2 = np.zeros(N_ANGLES)
            for k, d in enumerate(CAND_DIRS):
                om = torch.FloatTensor(
                    d * ACTION_SCALE_A2).unsqueeze(0).to(DEVICE)
                q_vals_a2[k] = net.value_estimator(h2_w, om).item()

            best_k = int(np.argmax(q_vals_a2))
            pref_U_a2[j, i] = CAND_DIRS[best_k, 0]
            pref_V_a2[j, i] = CAND_DIRS[best_k, 1]
            pref_S_a2[j, i] = q_vals_a2.max() - q_vals_a2.min()

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{GRID_SIZE} done")

print("Done.")

# =========================================================================
# Visualization
# =========================================================================
xx, yy = np.meshgrid(xs, ys)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(
    'Preferred Action Direction  (network reveals its own goal)\n'
    f'Arrows = direction that maximizes Q  |  '
    f'Color = preference strength (Q_max - Q_min over {N_ANGLES} directions)',
    fontsize=12
)

panels = [
    (axes[0], pref_U_a1, pref_V_a1, pref_S_a1,
     'A-1 Critic\n'
     'If learned: arrows converge toward Red (-9, +9)\n'
     'A-2 fixed at Green'),
    (axes[1], pref_U_a2, pref_V_a2, pref_S_a2,
     'A-2 ValueEstimator\n'
     'If learned: arrows converge toward Green (-9, -9)\n'
     f'A-1 fixed at Center, {WARMUP_STEPS}-step warmup'),
]

for ax, U, V, S, title in panels:
    # Background: preference strength
    im = ax.imshow(
        S,
        extent=[-9, 9, -9, 9],
        origin='lower',
        cmap='YlOrRd',
        vmin=0,
        vmax=max(S.max(), 1e-6),
        aspect='equal',
        alpha=0.6,
    )
    plt.colorbar(im, ax=ax, shrink=0.80,
                 label='Preference strength (Q_max - Q_min)')

    # Arrows: preferred direction at each grid point
    ax.quiver(
        xx, yy, U, V,
        scale=GRID_SIZE * 1.4,
        color='navy',
        alpha=0.75,
        width=0.004,
        headwidth=4,
        headlength=5,
    )

    # Landmarks
    for name, lx, ly, lc in LANDMARKS:
        ax.plot(lx, ly, '*', color=lc, markersize=16,
                markeredgecolor='black', markeredgewidth=0.6, label=name)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {SAVE_PATH}")

# =========================================================================
# Numeric summary: what fraction of arrows point toward each landmark?
# =========================================================================
def frac_toward(U, V, target_pos, grid_xs, grid_ys, threshold_deg=45):
    """Fraction of arrows pointing within threshold_deg of target direction."""
    count = 0
    total = 0
    for i, x in enumerate(grid_xs):
        for j, y in enumerate(grid_ys):
            pos = np.array([x, y])
            to_target = target_pos - pos
            dist = np.linalg.norm(to_target)
            if dist < 1e-3:
                continue
            to_target /= dist
            arrow = np.array([U[j, i], V[j, i]])
            dot = np.clip(np.dot(to_target, arrow), -1, 1)
            angle = np.degrees(np.arccos(dot))
            if angle < threshold_deg:
                count += 1
            total += 1
    return count / max(total, 1)

RED_POS   = np.array([-9.0,  9.0])
GREEN_POS = np.array([-9.0, -9.0])

print("\n=== Preferred Direction Summary ===")
print(f"  (fraction of arrows within 45° of each landmark)")
print(f"  {'':18s}  A-1 Critic   A-2 ValueEst")
for lname, lpos in [('Red  (-9,+9)', RED_POS),
                     ('Green(-9,-9)', GREEN_POS),
                     ('Blue ( 9,-9)', np.array([9.0, -9.0])),
                     ('Cyan ( 9,+9)', np.array([9.0,  9.0]))]:
    f1 = frac_toward(pref_U_a1, pref_V_a1, lpos, xs, ys)
    f2 = frac_toward(pref_U_a2, pref_V_a2, lpos, xs, ys)
    print(f"  {lname:18s}:  {f1*100:5.1f}%        {f2*100:5.1f}%")
