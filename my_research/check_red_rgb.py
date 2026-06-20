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

# 赤ランドマークに近い位置
env.self_agent.p = np.array([9.0, 9.0])
v, _, _, _, _ = env.step()

# 画像右端（赤が見えているあたり）のピクセルを確認
print("画像右端のピクセルRGB値（列55〜63）:")
for col in range(55, 64):
    for row in range(16):
        r, g, b = v[row, col, 0], v[row, col, 1], v[row, col, 2]
        if r > 0.5:
            print(f"  ({row},{col}): R={r:.3f}, G={g:.3f}, B={b:.3f}")
