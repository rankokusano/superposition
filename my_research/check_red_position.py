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
    'red_corner (9,9)': np.array([9.0, 9.0]),
    'center (0,0)': np.array([0.0, 0.0]),
    'opposite (-9,-9)': np.array([-9.0, -9.0]),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 6))

for idx, (name, pos) in enumerate(positions.items()):
    env.self_agent.p = pos
    v, _, _, _, _ = env.step()
    
    r = v[:, :, 0]
    g = v[:, :, 1]
    b = v[:, :, 2]
    red_mask = (r > 0.9) & (g < 0.05) & (b < 0.05)
    
    axes[0, idx].imshow(v)
    axes[0, idx].set_title(f'{name}\n赤ピクセル数: {red_mask.sum()}')
    
    # 赤マスクを可視化
    axes[1, idx].imshow(red_mask, cmap='Reds')
    axes[1, idx].set_title('赤ピクセルの位置')

plt.tight_layout()
plt.savefig('/work/my_research/red_position_check.png', dpi=100, bbox_inches='tight')
print("保存完了: red_position_check.png")
