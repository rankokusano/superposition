import sys
sys.path.append('/work')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation import creator
from simulation.util import load_config
from agent2_green import GreenFollowerAgent

config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
env = creator.create_environment(config.environment)
env.init()
env.off_display()
env.reset()

agent2 = GreenFollowerAgent()

# エージェント2をランダムな初期位置から動かす
trajectory = []
for _ in range(5):
    env.reset()
    pos = env.self_agent.p.copy()
    traj = [pos.copy()]

    for step in range(100):
        action = agent2.get_action(pos)
        pos = pos + action
        pos = np.clip(pos, -9.5, 9.5)
        traj.append(pos.copy())

    trajectory.append(np.array(traj))

# 軌跡を可視化
fig, axes = plt.subplots(1, 5, figsize=(25, 5), squeeze=False)
for ep, traj in enumerate(trajectory):
    ax = axes[0, ep]
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.plot(-9,  9, 'r*', markersize=12, label='Red')
    ax.plot(-9, -9, 'g*', markersize=12, label='Green（目標）')
    ax.plot( 9,  9, 'c*', markersize=8)
    ax.plot( 9, -9, 'b*', markersize=8)
    ax.plot(traj[:, 0], traj[:, 1], 'k-', linewidth=1)
    ax.plot(traj[0, 0], traj[0, 1], 'ko', markersize=8, label='Start')
    ax.plot(traj[-1, 0], traj[-1, 1], 'k^', markersize=8, label='End')
    ax.set_title(f'Episode {ep+1}')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

plt.suptitle('Agent2 Trajectory (Green Follower)', fontsize=14)
plt.tight_layout()
plt.savefig('/work/my_research/trajectory_agent2.png', dpi=100)
print("Saved: trajectory_agent2.png")
