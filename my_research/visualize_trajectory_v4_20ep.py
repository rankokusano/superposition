import sys
sys.path.append('/work')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import creator
from simulation.util import load_config
from rl_agent_discrete import DQNAgentDiscrete

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(v):
    v = np.transpose(v, (2, 0, 1))
    v = torch.FloatTensor(v).unsqueeze(0).to(DEVICE)
    return v

config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
env = creator.create_environment(config.environment)
env.init()
env.off_display()

model = DQNAgentDiscrete().to(DEVICE)
model.load_state_dict(torch.load('/work/my_research/rl_model_v4.pth'))
model.eval()

n_episodes = 20
fig, axes = plt.subplots(4, 5, figsize=(25, 20), squeeze=False)

for ep in range(n_episodes):
    env.reset()
    v_raw, _, _, _, _ = env.step()
    state = preprocess(v_raw)
    trajectory = [env.self_agent.p.copy()]

    for step in range(100):
        action, action_idx = model.get_action(state, epsilon=0.0)
        env.self_agent.p = env.self_agent.p + action
        env.self_agent.p = np.clip(env.self_agent.p, -9.5, 9.5)
        v_raw, _, _, _, _ = env.step()
        state = preprocess(v_raw)
        trajectory.append(env.self_agent.p.copy())

    trajectory = np.array(trajectory)
    ax = axes[ep // 5, ep % 5]
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.plot(-9,  9, 'r*', markersize=12, label='Red (+1)')
    ax.plot(-9, -9, 'g*', markersize=12, label='Green (-1)')
    ax.plot( 9,  9, 'c*', markersize=8)
    ax.plot( 9, -9, 'b*', markersize=8)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=1, alpha=0.5)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'ko', markersize=8, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'k^', markersize=8, label='End')
    ax.set_title(f'Episode {ep+1}')
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.3)

plt.suptitle('Agent Trajectory v4 (20 episodes, epsilon=0)', fontsize=14)
plt.tight_layout()
plt.savefig('/work/my_research/trajectory_v4_20ep.png', dpi=100, bbox_inches='tight')
print("Saved: trajectory_v4_20ep.png")
