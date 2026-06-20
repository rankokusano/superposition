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

fig, axes = plt.subplots(1, 5, figsize=(25, 5))

for ep in range(5):
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
    ax = axes[ep]
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.plot(-9,  9, 'r*', markersize=15, label='Red (+1)')
    ax.plot(-9, -9, 'g*', markersize=15, label='Green (-1)')
    ax.plot( 9,  9, 'c*', markersize=10, label='Cyan')
    ax.plot( 9, -9, 'b*', markersize=10, label='Blue')
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=1, alpha=0.5)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'ko', markersize=8, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'k^', markersize=8, label='End')
    ax.set_title(f'Episode {ep+1}')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

plt.suptitle('Agent Trajectory v4 (discrete, epsilon=0)', fontsize=14)
plt.tight_layout()
plt.savefig('/work/my_research/trajectory_v4.png', dpi=150, bbox_inches='tight')
print("Saved: trajectory_v4.png")
