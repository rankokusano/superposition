import sys
sys.path.append('/work')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation import creator
from simulation.util import load_config

config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
env = creator.create_environment(config.environment)
env.init()
env.off_display()
env.reset()

positions = {
    'red_corner': np.array([9.0, 9.0]),
    'center': np.array([0.0, 0.0]),
    'opposite': np.array([-9.0, -9.0]),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, pos) in enumerate(positions.items()):
    env.self_agent.p = pos
    v, _, _, _, _ = env.step()
    axes[idx].imshow(v)
    axes[idx].set_title(f'{name}\npos={pos}')

plt.savefig('/work/my_research/vision_check.png', dpi=100, bbox_inches='tight')
print("保存完了: vision_check.png")
