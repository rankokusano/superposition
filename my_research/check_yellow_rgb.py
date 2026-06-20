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

# 黄色ランドマークの真横
env.self_agent.p = np.array([9.0, 9.0])
v, _, _, _, _ = env.step()

print("黄色ランドマーク付近のピクセルRGB値（列30〜50）:")
for col in range(30, 50):
    for row in range(8):
        r, g, b = v[row, col, 0], v[row, col, 1], v[row, col, 2]
        if r != g or g != b:  # グレー以外のピクセル
            print(f"  ({row},{col}): R={r:.3f}, G={g:.3f}, B={b:.3f}")
