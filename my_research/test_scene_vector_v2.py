import sys
sys.path.insert(0, '/work')

import torch
import numpy as np
from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from model.model import SuperpositionNetworkMotionGenerationFeaturePrediction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# exp3のモデル設定を読み込む
model_config = load_exp_config(
    '/work/config/model/SuperpositionNetworkMotionGenerationFeaturePrediction/default.yml'
)

# モデルの初期化
model = SuperpositionNetworkMotionGenerationFeaturePrediction(
    model_config
).to(DEVICE)
model.load_state_dict(
    torch.load('/work/data/result/exp3/0/model/00200.pth',
               map_location=DEVICE),
    strict=False
)
model.eval()

# 環境の初期化
env_config = load_config(
    '/work/simulation/config/collect/self_random_other_stay.yml'
)
env = creator.create_environment(env_config.environment)
env.init()
env.off_display()
env.reset()

# 1ステップ実行
v_raw, sm, om, sp, op = env.step()

v = torch.FloatTensor(v_raw).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
sm_t = torch.FloatTensor(sm).unsqueeze(0).to(DEVICE)

model.superposition_module.init_state(1)
model.motion_generator_module.init_state(1)

with torch.no_grad():
    sv_enc = model.self_vision_encoder_module(v)
    ov_enc = model.other_vision_encoder_module(v)

    # Motion Generatorでom_tを生成
    om_generated = model.motion_generator_module(ov_enc)

    ss, os = model.superposition_module(
        sv_enc, sm_t, ov_enc, om_generated
    )

print(f"ss（自己の内部状態）のサイズ：{ss.shape}")
print(f"os（他者の内部状態）のサイズ：{os.shape}")
print(f"om_generated（生成された他者の運動）のサイズ：{om_generated.shape}")
print(f"om_generated の値：{om_generated[0]}")
print("成功！")
