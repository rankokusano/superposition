import sys
sys.path.append('/work')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import creator
from simulation.util import load_config
from rl_agent_sac import ActorLSTM, CriticLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(v):
    v = np.transpose(v, (2, 0, 1))
    v = torch.FloatTensor(v).unsqueeze(0).to(DEVICE)
    return v

# モデル読み込み
actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load('/work/my_research/rl_model_v6_actor.pth'))
actor.eval()

critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load('/work/my_research/rl_model_v6_critic.pth'))
critic.eval()

# 環境
config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
env = creator.create_environment(config.environment)
env.init()
env.off_display()

# A-1の位置をグリッドで変えてQ値を計算
# A-2は固定（中央に置く）
FIXED_A2_POS = np.array([0.0, 0.0])

grid_size = 20
xs = np.linspace(-9, 9, grid_size)
ys = np.linspace(-9, 9, grid_size)
q_map = np.zeros((grid_size, grid_size))

print("A-1のQ値マップ計算中...")
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        # A-1をその位置に置いて画像を取得
        a1_pos = np.array([x, y])
        v_raw = env.capture_at_pos(a1_pos, FIXED_A2_POS)
        v_t = preprocess(v_raw)

        with torch.no_grad():
            action, _, _ = actor.sample(v_t)
            q, _ = critic(v_t, action)
        q_map[j, i] = q.item()

plt.figure(figsize=(8, 7))
plt.imshow(
    q_map,
    extent=[-9, 9, -9, 9],
    origin='lower',
    cmap='RdYlGn',
    aspect='equal'
)
plt.colorbar(label='Q-value')
plt.title('A-1 Q-value Map\n(A-1 position varies, A-2 fixed at center)')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(-9,  9, 'r*', markersize=15, label='Red (+1)')
plt.plot(-9, -9, 'g*', markersize=15, label='Green (-1)')
plt.plot( 9,  9, 'c*', markersize=10, label='Cyan')
plt.plot( 9, -9, 'b*', markersize=10, label='Blue')
plt.legend()
plt.savefig('/work/my_research/q_map_a1_correct.png', dpi=150, bbox_inches='tight')
print("Saved: q_map_a1_correct.png")

# 数値確認
print("\n=== 重要座標でのQ値 ===")
red_i   = int(np.argmin(np.abs(xs - (-9))))
red_j   = int(np.argmin(np.abs(ys -  (9))))
green_i = int(np.argmin(np.abs(xs - (-9))))
green_j = int(np.argmin(np.abs(ys - (-9))))
center_i = grid_size // 2
center_j = grid_size // 2

print(f"Red(-9,9)    = {q_map[red_j, red_i]:.4f}")
print(f"Green(-9,-9) = {q_map[green_j, green_i]:.4f}")
print(f"Center(0,0)  = {q_map[center_j, center_i]:.4f}")
