"""
visualize_action_intent.py

Action Intent Map: Goal-directed preference (delta-Q)

delta_Q(pos) = Q(action toward goal) - Q(action away from goal)

  delta_Q > 0 : prefers moving toward goal at this position   <- desired
  delta_Q < 0 : prefers moving away from goal (not learned)

Left panel : A-1 Critic delta_Q  (goal = Red landmark at -9,+9)
Right panel: A-2 ValueEstimator delta_Q  (goal = Green landmark at -9,-9)

If both panels show delta_Q > 0 across most positions,
A-1's Process-2 has acquired A-2's behavioral intent.

Run:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_action_intent.py

Output:
    step2/action_intent_map.png
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
SAVE_PATH          = os.path.join(_HERE, 'action_intent_map.png')

GRID_SIZE    = 20
WARMUP_STEPS = 20   # steps to warm up h² before reading delta-Q

RED_POS   = np.array([-9.0,  9.0])   # A-1's goal
GREEN_POS = np.array([-9.0, -9.0])   # A-2's goal

A2_FIXED_POS = GREEN_POS              # A-2 fixed at green for A-1 panel
A1_FIXED_POS = np.array([0.0, 0.0])  # A-1 fixed at center for A-2 panel

ACTION_SCALE_A1 = 0.7   # within tanh range (Actor output scale)
ACTION_SCALE_A2 = 1.0   # GreenFollower speed

LANDMARKS = [
    ('Red',   -9,  9, 'red'),
    ('Green', -9, -9, 'limegreen'),
    ('Blue',   9, -9, 'blue'),
    ('Cyan',   9,  9, 'cyan'),
]

Q_SCALE = 1.0

# =========================================================================
def goal_direction(pos, goal):
    """Unit vector from pos toward goal. Returns north when at goal."""
    delta = goal - pos
    dist = np.linalg.norm(delta)
    if dist < 1e-6:
        return np.array([0.0, 1.0])
    return delta / dist


# =========================================================================
print(f"Device: {DEVICE}")

model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithQ(model_config).to(DEVICE)

if not os.path.exists(MODEL_SAVE_PATH):
    print("[WARNING] model_q.pth not found. Using Iizuka init.")
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

dq_a1 = np.zeros((GRID_SIZE, GRID_SIZE))   # A-1 Critic delta-Q
dq_a2 = np.zeros((GRID_SIZE, GRID_SIZE))   # A-2 ValueEstimator delta-Q

print(f"\nComputing action intent maps ({GRID_SIZE}x{GRID_SIZE}, "
      f"warmup={WARMUP_STEPS})...")

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        pos = np.array([x, y])

        with torch.no_grad():

            # ── A-1 panel: Critic delta-Q (goal = Red) ───────────────────
            # Visual scene: A-1 at (x,y), A-2 fixed at green
            v_a1 = env.capture_at_pos(pos, A2_FIXED_POS)
            v_t_a1 = torch.FloatTensor(
                v_a1.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            d = goal_direction(pos, RED_POS)
            a_toward_red = torch.FloatTensor(
                d * ACTION_SCALE_A1).unsqueeze(0).to(DEVICE)
            a_away_red = torch.FloatTensor(
                -d * ACTION_SCALE_A1).unsqueeze(0).to(DEVICE)

            q_toward, _ = critic(v_t_a1, a_toward_red)
            q_away,   _ = critic(v_t_a1, a_away_red)
            dq_a1[j, i] = (q_toward - q_away).item()

            # ── A-2 panel: ValueEstimator delta-Q (goal = Green) ─────────
            # Visual scene: A-1 fixed at center, A-2 at (x,y)
            v_a2 = env.capture_at_pos(A1_FIXED_POS, pos)
            v_t_a2 = torch.FloatTensor(
                v_a2.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            d2 = goal_direction(pos, GREEN_POS)
            a_toward_green = torch.FloatTensor(
                d2 * ACTION_SCALE_A2).unsqueeze(0).to(DEVICE)
            a_away_green = torch.FloatTensor(
                -d2 * ACTION_SCALE_A2).unsqueeze(0).to(DEVICE)

            # Warm up h² by simulating A-2 approaching green from pos
            net.superposition_module.init_state(1)
            actor_h = None
            for _ in range(WARMUP_STEPS):
                a_tmp, _, actor_h = actor.sample(v_t_a2, actor_h)
                q_s, _ = critic(v_t_a2, a_tmp)
                x_in = {
                    'self_vision':  v_t_a2,
                    'self_motion':  a_tmp,
                    'other_motion': a_toward_green,   # A-2 moving toward green
                }
                _, _, h2 = net(x_in, q_s / Q_SCALE, 0.0, 0.0)
                net.superposition_module.detach_state()

            h2_warmed = net.superposition_module.state['other'].hidden
            q_toward_ve = net.value_estimator(h2_warmed, a_toward_green)
            q_away_ve   = net.value_estimator(h2_warmed, a_away_green)
            dq_a2[j, i] = (q_toward_ve - q_away_ve).item()

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{GRID_SIZE} done")

print("Done.")

# =========================================================================
# Visualization
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    'Action Intent: Preference for Goal-Directed Motion\n'
    r'$\Delta Q$ = Q(toward goal) - Q(away from goal)'
    '   |   red=prefer goal, blue=prefer opposite',
    fontsize=12
)

panels = [
    (axes[0], dq_a1, 'A-1 Critic  ΔQ\n'
                      'Goal: Red (-9,+9)\n'
                      'A-2 fixed at Green',
     RED_POS),
    (axes[1], dq_a2, f'A-2 ValueEstimator  ΔQ\n'
                      f'Goal: Green (-9,-9)  [{WARMUP_STEPS}-step warmup]\n'
                      f'A-1 fixed at Center',
     GREEN_POS),
]

for ax, dq, title, goal in panels:
    vabs = max(np.abs(dq).max(), 1e-6)
    im = ax.imshow(
        dq,
        extent=[-9, 9, -9, 9],
        origin='lower',
        cmap='RdBu_r',       # red=positive=prefers goal, blue=negative
        vmin=-vabs,
        vmax=vabs,
        aspect='equal',
    )
    plt.colorbar(im, ax=ax, shrink=0.85,
                 label='ΔQ  (positive = prefers toward goal)')

    for name, lx, ly, lc in LANDMARKS:
        ax.plot(lx, ly, '*', color=lc, markersize=14,
                markeredgecolor='black', markeredgewidth=0.5, label=name)

    # Mark the goal with a bold outline
    ax.plot(goal[0], goal[1], 'o', color='white', markersize=20,
            markeredgecolor='gold', markeredgewidth=2.5, zorder=3,
            alpha=0.6, label='Goal')

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
# Numeric summary
# =========================================================================

pct_positive_a1 = float((dq_a1 > 0).mean() * 100)
pct_positive_a2 = float((dq_a2 > 0).mean() * 100)

print("\n=== Action Intent Summary ===")
print(f"  A-1 Critic:          {pct_positive_a1:.1f}% of positions prefer toward-Red")
print(f"  A-2 ValueEstimator:  {pct_positive_a2:.1f}% of positions prefer toward-Green")
print(f"  (100% = perfect goal-directed intent, 50% = random)")

def idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))

print("\n  Delta-Q at key positions:")
print(f"  {'Position':>20}  A-1 Critic    A-2 ValueEst")
for label, px, py in [('Red  (-9,+9)',  -9,  9),
                       ('Green(-9,-9)', -9, -9),
                       ('Center(0,0)',   0,  0),
                       ('Blue ( 9,-9)',  9, -9)]:
    v1 = dq_a1[idx(ys, py), idx(xs, px)]
    v2 = dq_a2[idx(ys, py), idx(xs, px)]
    print(f"  {label:>20}:  {v1:+.4f}        {v2:+.4f}")
