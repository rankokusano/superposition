import sys
sys.path.append('/work')
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation import creator
from simulation.util import load_config
from rl_agent import DQNAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(v):
    v = np.transpose(v, (2, 0, 1))
    v = torch.FloatTensor(v).unsqueeze(0).to(DEVICE)
    return v

config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
env = creator.create_environment(config.environment)
env.init()
env.off_display()

model = DQNAgent().to(DEVICE)
model.load_state_dict(torch.load('/work/my_research/rl_model_v3.pth'))
model.eval()

grid_size = 20
xs = np.linspace(-9, 9, grid_size)
ys = np.linspace(-9, 9, grid_size)
q_map = np.zeros((grid_size, grid_size))

print("Q値マップ計算中...")
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        pos = np.array([x, y])
        v = env.capture_at_pos(pos, np.array([0.0, 0.0]))
        state = preprocess(v)
        with torch.no_grad():
            q, a = model(state)
        q_map[j, i] = q.item()

plt.figure(figsize=(8, 7))
plt.imshow(q_map, extent=[-9, 9, -9, 9], origin='lower', cmap='RdYlGn', aspect='equal')
plt.colorbar(label='Q-value')
plt.title('Q-value Map v3')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(-9,  9, 'r*', markersize=15, label='Red (+1)')
plt.plot(-9, -9, 'g*', markersize=15, label='Green (-1)')
plt.legend()
plt.savefig('/work/my_research/q_value_map_v3.png', dpi=150, bbox_inches='tight')
print("Saved: q_value_map_v3.png")
