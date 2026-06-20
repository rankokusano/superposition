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
    v = v.astype(np.float32) / 255.0
    v = np.transpose(v, (2, 0, 1))
    v = torch.FloatTensor(v).unsqueeze(0).to(DEVICE)
    return v

def visualize():
    # モデルのロード
    model = DQNAgent().to(DEVICE)
    model.load_state_dict(torch.load('/work/my_research/rl_model.pth'))
    model.eval()

    # 環境の初期化
    config = load_config('/work/simulation/config/collect/self_random_other_stay.yml')
    env = creator.create_environment(config.environment)
    env.init()
    env.off_display()

    # アリーナ全体のグリッドでQ-valueを計算
    # 飯塚さんの2次元カラーマップと同じ形式
    grid_size = 20
    xs = np.linspace(-9, 9, grid_size)
    ys = np.linspace(-9, 9, grid_size)
    q_map = np.zeros((grid_size, grid_size))

    print("Calculating Q-value map...")
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # エージェントをその位置に置いて画像を取得
            pos = np.array([x, y])
            v = env.capture_at_pos(pos, np.array([0.0, 0.0]))
            state = preprocess(v)

            with torch.no_grad():
                q, a = model(state)
            q_map[j, i] = q.item()

    # 2次元カラーマップとして表示
    plt.figure(figsize=(8, 7))
    plt.imshow(
        q_map,
        extent=[-9, 9, -9, 9],
        origin='lower',
        cmap='hot',
        aspect='equal'
    )
    plt.colorbar(label='Q-value')
    plt.title('Q-value Map (Red object: top-right)')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 赤いオブジェクトの位置を表示
    plt.plot(9, 9, 'b*', markersize=15, label='Red object')
    plt.legend()

    plt.savefig('/work/my_research/q_value_map.png', dpi=150, bbox_inches='tight')
    print("Saved: q_value_map.png")

if __name__ == "__main__":
    visualize()
