import sys
import os
sys.path.insert(0, '/work')
sys.path.insert(0, '/work/my_research')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import creator
from simulation.util import load_config
from rl_agent_sac import ActorLSTM, CriticLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(v):
    v = np.transpose(v, (2, 0, 1))
    v = torch.FloatTensor(v).unsqueeze(0).to(DEVICE)
    return v

actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load('/work/my_research/rl_model_v6_actor.pth'))
actor.eval()

critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load('/work/my_research/rl_model_v6_critic.pth'))
critic.eval()

config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
env = creator.create_environment(config.environment)
env.init()
env.off_display()

all_positions = []
all_q_values = []

n_episodes = 20
for ep in range(n_episodes):
    env.reset()
    v_raw, _, _, sp, _ = env.step()
    state = preprocess(v_raw)
    hidden = None

    for step in range(100):
        with torch.no_grad():
            action, _, hidden = actor.sample(state, hidden)
            q, _ = critic(state, action)

        all_positions.append(sp.copy())
        all_q_values.append(q.item())

        action_np = action.squeeze(0).cpu().numpy()
        env.self_agent.p = env.self_agent.p + action_np
        env.self_agent.p = np.clip(env.self_agent.p, -9.5, 9.5)
        v_raw, _, _, sp, _ = env.step()
        state = preprocess(v_raw)

positions = np.array(all_positions)
q_values = np.array(all_q_values)

plt.figure(figsize=(8, 7))
scatter = plt.scatter(
    positions[:, 0], positions[:, 1],
    c=q_values, cmap='RdYlGn',
    s=10, alpha=0.6
)
plt.colorbar(scatter, label='Q-value')
plt.title('A-1 Q-value Map\n(actual trajectory, 20 episodes)')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.plot(-9,  9, 'r*', markersize=15, label='Red (+1)')
plt.plot(-9, -9, 'g*', markersize=15, label='Green (-1)')
plt.plot( 9,  9, 'c*', markersize=10, label='Cyan')
plt.plot( 9, -9, 'b*', markersize=10, label='Blue')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/work/my_research/step2/q_map_a1_trajectory.png', dpi=150, bbox_inches='tight')
print("Saved: q_map_a1_trajectory.png")
print(f"Q値の範囲: min={q_values.min():.4f}, max={q_values.max():.4f}")
