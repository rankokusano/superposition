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

def get_dominant_colors(v):
    r = v[:,:,0]
    g = v[:,:,1]
    b = v[:,:,2]
    
    red_count    = int(((r > 0.9) & (g < 0.1) & (b < 0.1)).sum())
    green_count  = int(((g > 0.6) & (r < 0.1) & (b < 0.1)).sum())
    blue_count   = int(((b > 0.6) & (r < 0.1) & (g < 0.1)).sum())
    yellow_count = int(((r > 0.6) & (g > 0.6) & (b < 0.1)).sum())
    cyan_count   = int(((g > 0.6) & (b > 0.6) & (r < 0.1)).sum())
    
    return {
        'red': red_count,
        'green': green_count,
        'blue': blue_count,
        'yellow': yellow_count,
        'cyan': cyan_count,
    }

positions = {
    'right-bottom (9,-9)':  np.array([9.0, -9.0]),
    'right-top (9,9)':      np.array([9.0,  9.0]),
    'left-top (-9,9)':      np.array([-9.0,  9.0]),
    'left-bottom (-9,-9)':  np.array([-9.0, -9.0]),
    'center (0,0)':         np.array([0.0,  0.0]),
}

for name, pos in positions.items():
    env.self_agent.p = pos
    v, _, _, _, _ = env.step()
    colors = get_dominant_colors(v)
    print(f"{name}:")
    for color, count in colors.items():
        if count > 0:
            print(f"  {color}: {count}px")
