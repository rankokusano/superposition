import sys
sys.path.append('/work')
import numpy as np
from simulation import creator
from simulation.util import load_config

def get_reward_from_vision(v_raw):
    r = v_raw[:, :, 0]
    g = v_raw[:, :, 1]
    b = v_raw[:, :, 2]
    red_mask = (r > 0.9) & (g < 0.05) & (b < 0.05)
    return float(red_mask.sum())

config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
env = creator.create_environment(config.environment)
env.init()
env.off_display()
env.reset()

# 赤ランドマークは左上(-9, 9)
env.self_agent.p = np.array([-9.0, 9.0])
v, _, _, _, _ = env.step()
print(f"赤の真横(-9,9)にいるとき：報酬 = {get_reward_from_vision(v):.4f}")

# 真ん中
env.self_agent.p = np.array([0.0, 0.0])
v, _, _, _, _ = env.step()
print(f"真ん中(0,0)にいるとき：報酬 = {get_reward_from_vision(v):.4f}")

# 反対側（右下）
env.self_agent.p = np.array([9.0, -9.0])
v, _, _, _, _ = env.step()
print(f"反対側(9,-9)にいるとき：報酬 = {get_reward_from_vision(v):.4f}")
