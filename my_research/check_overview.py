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

# 俯瞰カメラに切り替え
env.world.set_camera('overview')
v = env.world.capture()

plt.figure(figsize=(6,6))
plt.imshow(v)
plt.title('Overview Map')
plt.savefig('/work/my_research/overview_map.png', dpi=100)
print("保存完了")
