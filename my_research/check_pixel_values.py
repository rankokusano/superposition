import sys
sys.path.append('/work')
import numpy as np
from simulation import creator
from simulation.util import load_config

config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
env = creator.create_environment(config.environment)
env.init()
env.off_display()
env.reset()

# 緑の真横
env.self_agent.p = np.array([-9.0, -9.0])
v, _, _, _, _ = env.step()
g = v[:,:,1]
r = v[:,:,0]
b = v[:,:,2]

print("緑ピクセルの閾値別カウント：")
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    count = int(((g > threshold) & (r < 0.1) & (b < 0.1)).sum())
    print(f"  g > {threshold}：{count}px")

print("\n赤ピクセルの閾値別カウント：")
env.self_agent.p = np.array([-9.0, 9.0])
v, _, _, _, _ = env.step()
r2 = v[:,:,0]
g2 = v[:,:,1]
b2 = v[:,:,2]
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    count = int(((r2 > threshold) & (g2 < 0.1) & (b2 < 0.1)).sum())
    print(f"  r > {threshold}：{count}px")
