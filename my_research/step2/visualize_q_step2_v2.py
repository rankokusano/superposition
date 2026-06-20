import sys
import os
sys.path.insert(0, '/work/my_research/step2')
sys.path.insert(0, '/work')
sys.path.insert(0, '/work/my_research')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_green import GreenFollowerAgent
from model_q import SuperpositionNetworkWithQ

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_SCALE = 50.0

def preprocess(v):
    v = np.transpose(v, (2, 0, 1))
    v = torch.FloatTensor(v).unsqueeze(0).to(DEVICE)
    return v

# モデル読み込み
actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load('/work/my_research/rl_model_v6_actor.pth'))
actor.eval()

critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load('/work/my_research/rl_model_v6_critic.pth'))
critic.eval()

model_config = load_exp_config(
    '/work/config/model/SuperpositionNetworkFeaturePrediction/default.yml'
)
net = SuperpositionNetworkWithQ(model_config).to(DEVICE)
ckpt = torch.load('/work/my_research/step2/model_q.pth', map_location=DEVICE)
net.load_state_dict(ckpt['net_state_dict'])
net.eval()

config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
env = creator.create_environment(config.environment)
env.init()
env.off_display()

green_follower = GreenFollowerAgent()

# A-1とA-2の軌跡＋ValueEstimatorのQ値を記録
n_episodes = 20
all_a1_pos = []
all_a2_pos = []
all_ve_q   = []

print("軌跡とQ値を計算中...")
for ep in range(n_episodes):
    env.reset()
    v_raw, _, _, sp, op = env.step()
    a1_pos = sp.copy()
    a2_pos = op.copy()

    actor_hidden = None
    critic_hidden = None
    net.superposition_module.init_state(1)

    for step in range(100):
        state = preprocess(v_raw)

        with torch.no_grad():
            # A-1の行動
            action, _, actor_hidden = actor.sample(state, actor_hidden)
            q_self, critic_hidden = critic(state, action, critic_hidden)
            q_self_scaled = q_self / Q_SCALE

            # A-2の行動
            om_np = green_follower.get_action(a2_pos)
            om_t = torch.FloatTensor(om_np).unsqueeze(0).to(DEVICE)
            sm_t = action.squeeze(0)

            # ValueEstimatorのQ値
            x_in = {
                'self_vision':  state,
                'self_motion':  sm_t.unsqueeze(0),
                'other_motion': om_t,
            }
            _, _, os_ = net(x_in, q_self_scaled, 0.0, 0.0)
            ve_q = net.value_estimator(os_, om_t)

        all_a1_pos.append(a1_pos.copy())
        all_a2_pos.append(a2_pos.copy())
        all_ve_q.append(ve_q.item())

        # 位置を更新
        action_np = action.squeeze(0).cpu().numpy()
        a1_pos = np.clip(a1_pos + action_np, -9.5, 9.5)
        a2_pos = np.clip(a2_pos + om_np, -9.5, 9.5)
        env.self_agent.p  = a1_pos
        env.other_agent.p = a2_pos
        v_raw, _, _, sp, op = env.step()
        a1_pos = sp.copy()
        a2_pos = op.copy()

a1_pos_arr = np.array(all_a1_pos)
a2_pos_arr = np.array(all_a2_pos)
ve_q_arr   = np.array(all_ve_q)

print(f"ValueEstimator Q値範囲: min={ve_q_arr.min():.4f}, max={ve_q_arr.max():.4f}")

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左：A-1の軌跡（ValueEstimatorのQ値で色付け）
sc1 = axes[0].scatter(
    a1_pos_arr[:, 0], a1_pos_arr[:, 1],
    c=ve_q_arr, cmap='RdYlGn', s=10, alpha=0.6
)
plt.colorbar(sc1, ax=axes[0], label='ValueEstimator Q-value')
axes[0].set_title('A-1の軌跡\n（ValueEstimatorのQ値で色付け）')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_xlim(-10, 10)
axes[0].set_ylim(-10, 10)
axes[0].plot(-9,  9, 'r*', markersize=15, label='Red (+1)')
axes[0].plot(-9, -9, 'g*', markersize=15, label='Green (-1)')
axes[0].plot( 9,  9, 'c*', markersize=10)
axes[0].plot( 9, -9, 'b*', markersize=10)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右：A-2の軌跡（ValueEstimatorのQ値で色付け）
sc2 = axes[1].scatter(
    a2_pos_arr[:, 0], a2_pos_arr[:, 1],
    c=ve_q_arr, cmap='RdYlGn', s=10, alpha=0.6
)
plt.colorbar(sc2, ax=axes[1], label='ValueEstimator Q-value')
axes[1].set_title('A-2の軌跡\n（ValueEstimatorのQ値で色付け）')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_xlim(-10, 10)
axes[1].set_ylim(-10, 10)
axes[1].plot(-9,  9, 'r*', markersize=15, label='Red (+1)')
axes[1].plot(-9, -9, 'g*', markersize=15, label='Green (-1)')
axes[1].plot( 9,  9, 'c*', markersize=10)
axes[1].plot( 9, -9, 'b*', markersize=10)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Step2: A-1・A-2の軌跡とValueEstimator Q値\n(20 episodes)', fontsize=13)
plt.tight_layout()
plt.savefig('/work/my_research/step2/trajectory_step2.png', dpi=150, bbox_inches='tight')
print("Saved: trajectory_step2.png")
