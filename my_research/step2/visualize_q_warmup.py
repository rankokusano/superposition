"""
visualize_q_warmup.py

Step2 学習結果の可視化（20ステップウォームアップ版）
─────────────────────────────────────────────────
静的Q値マップの限界を克服するため、各グリッド点で
A-2を20ステップ固定配置してh²を蓄積してからQ値を読む。

【左パネル】ValueEstimator Q̂: A-2の各位置でのQ推測
    A-2を(x,y)に固定し、20ステップ同じ視覚入力を流して
    h²を充分に育ててからQ値を読む。
    → 「この位置に長くいた後、ValueEstimatorはどう判断するか」

【右パネル】A-1 Critic Q: A-1の各位置でのQ値（比較用）

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_q_warmup.py

出力:
    step2/q_map_warmup.png
─────────────────────────────────────────────────
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
from agent2_green import GreenFollowerAgent
from model_q import SuperpositionNetworkWithQ

# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH    = os.path.join(_HERE, 'model_q.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')
SAVE_PATH          = os.path.join(_HERE, 'q_map_warmup.png')

GRID_SIZE    = 20
WARMUP_STEPS = 20   # h²を育てるステップ数

LANDMARKS = [
    ('Red',   -9,  9, 'red'),
    ('Green', -9, -9, 'limegreen'),
    ('Blue',   9, -9, 'blue'),
    ('Cyan',   9,  9, 'cyan'),
]

Q_SCALE = 1.0

# =========================================================================
print(f"Device: {DEVICE}")

model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithQ(model_config).to(DEVICE)

if not os.path.exists(MODEL_SAVE_PATH):
    print("[WARNING] model_q.pth not found. Using random init.")
    iizuka_ckpt = torch.load(
        os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth'),
        map_location=DEVICE)
    net.load_iizuka_weights(iizuka_ckpt, DEVICE)
else:
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt['net_state_dict'])
    Q_SCALE = ckpt.get('q_scale', 1.0)
    print(f"Loaded model: episodes={ckpt.get('episodes','?')}, "
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

green_follower = GreenFollowerAgent()

# =========================================================================
xs = np.linspace(-9, 9, GRID_SIZE)
ys = np.linspace(-9, 9, GRID_SIZE)

q_map_ve     = np.zeros((GRID_SIZE, GRID_SIZE))
q_map_critic = np.zeros((GRID_SIZE, GRID_SIZE))

A1_FIXED_POS = np.array([0.0, 0.0])
A2_FIXED_FOR_CRITIC = np.array([-9.0, -9.0])

print(f"\nComputing Q maps ({GRID_SIZE}x{GRID_SIZE} grid, warmup={WARMUP_STEPS} steps)...")

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        a2_pos = np.array([x, y])
        om_np  = green_follower.get_action(a2_pos)
        om_t   = torch.FloatTensor(om_np).unsqueeze(0).to(DEVICE)

        v_raw = env.capture_at_pos(A1_FIXED_POS, a2_pos)
        v_t   = torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # ── ValueEstimator with warmup ──────────────────────────────
            # A-2 stays at (x,y), A-1 stays at (0,0)
            # Run WARMUP_STEPS with same input to let h² accumulate context
            net.superposition_module.init_state(1)
            actor_h = None

            for _ in range(WARMUP_STEPS):
                a_tmp, _, actor_h = actor.sample(v_t, actor_h)
                q_s, _ = critic(v_t, a_tmp)
                q_s_scaled = q_s / Q_SCALE
                x_in = {
                    'self_vision':  v_t,
                    'self_motion':  a_tmp,
                    'other_motion': om_t,
                }
                _, _, h2 = net(x_in, q_s_scaled, 0.0, 0.0)
                net.superposition_module.detach_state()

            q_ve = net.value_estimator(h2, om_t)
            q_map_ve[j, i] = q_ve.item()

            # ── A-1 Critic ──────────────────────────────────────────────
            v_raw_a1 = env.capture_at_pos(a2_pos, A2_FIXED_FOR_CRITIC)
            v_t_a1   = torch.FloatTensor(
                v_raw_a1.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            a_ref, _, _ = actor.sample(v_t_a1)
            q_c, _ = critic(v_t_a1, a_ref)
            q_map_critic[j, i] = q_c.item()

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{GRID_SIZE} done")

print("Done.")

# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(
    f'Step2: Q-value Maps (warmup={WARMUP_STEPS} steps)\n'
    'Left: Process-2 ValueEstimator Q - should peak near Green (-9,-9)\n'
    'Right: A-1 Critic Q - reference, should peak near Red (-9,+9)',
    fontsize=11
)

panels = [
    (axes[0], q_map_ve,
     f'ValueEstimator Q(s,a) [Process-2]\n'
     f'A-2 fixed at each position, {WARMUP_STEPS}-step warmup\n'
     'Green peak = success'),
    (axes[1], q_map_critic,
     'A-1 Critic Q(s,a) [Reference]\n'
     'A-1 at each position, A-2 at green\n'
     'Red peak = expected'),
]

for ax, qmap, title in panels:
    vabs = max(np.abs(qmap).max(), 1e-6)
    im = ax.imshow(
        qmap,
        extent=[-9, 9, -9, 9],
        origin='lower',
        cmap='RdYlGn',
        vmin=-vabs,
        vmax=vabs,
        aspect='equal',
    )
    plt.colorbar(im, ax=ax, shrink=0.85, label='Q value')

    for name, lx, ly, lc in LANDMARKS:
        ax.plot(lx, ly, '*', color=lc, markersize=14,
                markeredgecolor='black', markeredgewidth=0.5, label=name)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\nSaved: {SAVE_PATH}")

# =========================================================================
def idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))

print("\n=== Q values at key positions ===")
print(f"{'':28s}  Green(-9,-9)  Red(-9,+9)  Center(0,0)")
for name, qmap in [('ValueEstimator (warmup)', q_map_ve),
                   ('A-1 Critic             ', q_map_critic)]:
    q_g = qmap[idx(ys, -9), idx(xs, -9)]
    q_r = qmap[idx(ys,  9), idx(xs, -9)]
    q_c = qmap[idx(ys,  0), idx(xs,  0)]
    print(f"  {name}:  {q_g:+.4f}        {q_r:+.4f}      {q_c:+.4f}")

# Check if green is the maximum
max_idx = np.unravel_index(np.argmax(q_map_ve), q_map_ve.shape)
max_x = xs[max_idx[1]]
max_y = ys[max_idx[0]]
print(f"\n  ValueEstimator max position: ({max_x:.1f}, {max_y:.1f})  "
      f"(should be near -9,-9 for success)")
