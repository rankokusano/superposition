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

def count_green(v):
    g = v[:,:,1]
    r = v[:,:,0]
    b = v[:,:,2]
    return int(((g > 0.6) & (r < 0.1) & (b < 0.1)).sum())

positions = {
    '緑の真横(-9,-9)': np.array([-9.0, -9.0]),
    '真ん中(0,0)':     np.array([0.0,  0.0]),
    '反対側(9,9)':     np.array([9.0,   9.0]),
}

for name, pos in positions.items():
    env.self_agent.p = pos
    v, _, _, _, _ = env.step()
    print(f"{name}：緑ピクセル数 = {count_green(v)}")
