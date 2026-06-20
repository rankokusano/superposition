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

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 赤の真横
env.self_agent.p = np.array([-9.0, 9.0])
v, _, _, _, _ = env.step()
axes[0].imshow(v)
r = v[:,:,0]; g = v[:,:,1]; b = v[:,:,2]
red_px = int(((r>0.9)&(g<0.05)&(b<0.05)).sum())
axes[0].set_title(f'Red landmark (-9,9)\nred pixels: {red_px}')

# 緑の真横
env.self_agent.p = np.array([-9.0, -9.0])
v, _, _, _, _ = env.step()
axes[1].imshow(v)
g2 = v[:,:,1]; r2 = v[:,:,0]; b2 = v[:,:,2]
green_px = int(((g2>0.6)&(r2<0.1)&(b2<0.1)).sum())
axes[1].set_title(f'Green landmark (-9,-9)\ngreen pixels: {green_px}')

plt.tight_layout()
plt.savefig('/work/my_research/pixel_diff_check.png', dpi=100)
print("保存完了")
