import sys
sys.path.insert(0, '/work')

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from model.model import SuperpositionNetworkMotionGenerationFeaturePrediction
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_green import GreenFollowerAgent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(v):
    v = np.transpose(v, (2, 0, 1))
    v = torch.FloatTensor(v).unsqueeze(0).to(DEVICE)
    return v

# 飯塚さんのモデル（exp3）を読み込む
model_config = load_exp_config(
    '/work/config/model/SuperpositionNetworkMotionGenerationFeaturePrediction/default.yml'
)
iizuka_model = SuperpositionNetworkMotionGenerationFeaturePrediction(
    model_config
).to(DEVICE)
iizuka_model.load_state_dict(
    torch.load('/work/data/result/exp3/0/model/00200.pth',
               map_location=DEVICE),
    strict=False
)
iizuka_model.eval()

# A-1のRL（学習済み・固定）
actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(
    torch.load('/work/my_research/rl_model_v6_actor.pth')
)
actor.eval()

critic1 = CriticLSTM().to(DEVICE)
critic1.load_state_dict(
    torch.load('/work/my_research/rl_model_v6_critic.pth')
)
critic1.eval()

# 環境の初期化
env_config = load_config(
    '/work/simulation/config/collect/self_random_other_stay.yml'
)
env = creator.create_environment(env_config.environment)
env.init()
env.off_display()

agent2 = GreenFollowerAgent()

# アリーナ全体でQ値を計算
grid_size = 20
xs = np.linspace(-9, 9, grid_size)
ys = np.linspace(-9, 9, grid_size)
q_map = np.zeros((grid_size, grid_size))

print("Q値マップ計算中...")
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        # A-2をその位置に置く
        pos = np.array([x, y])

        # A-2の行動を取得
        action_a2 = agent2.get_action(pos)

        # 環境からv_tを取得
        v = env.capture_at_pos(pos, np.array([0.0, 0.0]))
        state = preprocess(v)

        # 飯塚モデルからf²_m,tを生成
        iizuka_model.superposition_module.init_state(1)
        iizuka_model.motion_generator_module.init_state(1)

        with torch.no_grad():
            sv_enc = iizuka_model.self_vision_encoder_module(state)
            ov_enc = iizuka_model.other_vision_encoder_module(state)
            om_generated = iizuka_model.motion_generator_module(ov_enc)

            # f²_m,tをA-1のCriticに入力してQ値を得る
            q, _ = critic1(state, om_generated)

        q_map[j, i] = q.item()

# Q値マップを可視化
plt.figure(figsize=(8, 7))
plt.imshow(
    q_map,
    extent=[-9, 9, -9, 9],
    origin='lower',
    cmap='RdYlGn',
    aspect='equal'
)
plt.colorbar(label='Q-value')
plt.title('Q-value Map (A-2 motion via Motion Generator)')
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(-9,  9, 'r*', markersize=15, label='Red (+1 for A-1)')
plt.plot(-9, -9, 'g*', markersize=15, label='Green (target for A-2)')
plt.legend()
plt.savefig('/work/my_research/q_map_step2.png', dpi=150, bbox_inches='tight')
print("Saved: q_map_step2.png")
