"""
verify_iizuka_as_phase1.py

飯塚モデル (exp1_l1/0/model/00200.pth) を Phase 1 モデルとして使い、
h¹/h² の位置符号化を検証する。

【目的】
    我々の Phase 1 訓練 (SuperpositionNetworkPhase1) では
    h² が A-2 の位置を符号化できなかった。
    飯塚モデルは exp1 として十分な訓練を受けており、
    元々 Phase 1 相当の役割を担うモデルである。
    これが我々のシミュレーション上でも位置符号化できているかを検証する。

【重要】
    飯塚モデルの SM は Q 値なし (input=66 dim)。
    SuperpositionNetworkFeaturePrediction を使い、元の checkpoint をそのまま読む。

実行コマンド（コンテナ内）:
    cd /work
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/phase1/verify_iizuka_as_phase1.py \\
    2>&1 | tee my_research/step3/results/verify_iizuka.txt

出力: my_research/step3/results/verify_iizuka_phase1.png
"""

import sys, os
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

_HERE       = os.path.dirname(os.path.abspath(__file__))
STEP3_DIR   = os.path.abspath(os.path.join(_HERE, '..'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP3_DIR, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from util import scale_vision
from rl_agent_sac import ActorLSTM, CriticLSTM
from model.model import SuperpositionNetworkFeaturePrediction

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')
SAVE_PATH          = os.path.join(STEP3_DIR, 'results', 'verify_iizuka_phase1.png')

os.makedirs(os.path.join(STEP3_DIR, 'results'), exist_ok=True)

N_EPISODES = 50
MAX_STEPS  = 80
WORLD_HALF = 9.5

# =========================================================================
# モデル読み込み
# =========================================================================
print("飯塚モデル読み込み中...")
model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkFeaturePrediction(model_config).to(DEVICE)

ckpt = torch.load(IIZUKA_MODEL_PATH, map_location=DEVICE)
state_dict = ckpt.get('model', ckpt)
net.load_state_dict(state_dict)
net.eval()
for p in net.parameters():
    p.requires_grad = False
print(f"  loaded: {IIZUKA_MODEL_PATH}")

actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()

critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
critic.eval()

env_config = load_config(ENV_CONFIG_PATH)
env = creator.create_environment(env_config.environment)
env.init(); env.off_display()

Q_SCALE = 50.0

# =========================================================================
# データ収集（評価時は p_mask=1.0 固定: 元論文の exp1 評価と同様）
# =========================================================================
print(f"\nデータ収集中 ({N_EPISODES} ep × {MAX_STEPS} steps, p_mask=1.0 for eval)...")

all_h1        = []
all_h2        = []
all_self_pos  = []
all_other_pos = []

for ep in range(N_EPISODES):
    env.reset()
    net.init_state(1)
    actor_h = critic_h = None
    v_raw, _, _, self_pos, other_pos = env.step()
    other_pos_fixed = other_pos.copy()

    for step in range(MAX_STEPS):
        v_raw_t = torch.FloatTensor(
            v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        v_t = torch.FloatTensor(
            scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action, _, actor_h = actor.sample(v_raw_t, actor_h)
            q_raw, critic_h    = critic(v_raw_t, action, critic_h)

            p_mask = 0.0 if step == 0 else 1.0
            pred = net(
                {'self_vision': v_t, 'self_motion': action},
                p_mask_vision_self=p_mask,
                p_mask_vision_other=p_mask,
            )

            # 飯塚モデルの h¹/h² は superposition_module の state から取得
            state = net.superposition_module.state
            h1 = state['self'].hidden   # (1, 128)
            h2 = state['other'].hidden  # (1, 128)
            net.detach_state()

        all_h1.append(h1.squeeze(0).cpu().numpy())
        all_h2.append(h2.squeeze(0).cpu().numpy())
        all_self_pos.append(self_pos.copy())
        all_other_pos.append(other_pos_fixed.copy())

        a_np     = action.squeeze(0).cpu().numpy()
        self_pos = np.clip(self_pos + a_np, -9.5, 9.5)
        env.self_agent.p  = self_pos
        env.other_agent.p = other_pos_fixed
        v_raw, _, _, _, _ = env.step()

H1       = np.array(all_h1)
H2       = np.array(all_h2)
SelfPos  = np.array(all_self_pos)
OtherPos = np.array(all_other_pos)
N_total  = len(H1)
print(f"合計: {N_total} ステップ  (hidden_dim={H1.shape[1]})")

# =========================================================================
# 位置 → カラー変換
# =========================================================================
def pos_to_color(positions):
    nx = np.clip((positions[:, 0] + WORLD_HALF) / (2 * WORLD_HALF), 0, 1)
    ny = np.clip((positions[:, 1] + WORLD_HALF) / (2 * WORLD_HALF), 0, 1)
    return np.stack([ny, nx + ny - 2*nx*ny, (1-nx)*(1-ny)], axis=1)

# =========================================================================
# 検証 1: PCA 位置マップ
# =========================================================================
print("\n=== 検証 1: h¹/h² 位置マップ ===")
pca1 = PCA(n_components=2); H1_2d = pca1.fit_transform(H1)
pca2 = PCA(n_components=2); H2_2d = pca2.fit_transform(H2)
var1 = pca1.explained_variance_ratio_
var2 = pca2.explained_variance_ratio_
print(f"h¹ PCA 寄与率: PC1={var1[0]:.3f} PC2={var1[1]:.3f} 合計={var1.sum():.3f}")
print(f"h² PCA 寄与率: PC1={var2[0]:.3f} PC2={var2[1]:.3f} 合計={var2.sum():.3f}")

colors_self  = pos_to_color(SelfPos)
colors_other = pos_to_color(OtherPos)

# =========================================================================
# 検証 2: 線形回帰 R²
# =========================================================================
print("\n=== 検証 2: 線形回帰 R² ===")
H1_tr, H1_te, SP_tr, SP_te = train_test_split(H1, SelfPos,  test_size=0.3, random_state=42)
H2_tr, H2_te, OP_tr, OP_te = train_test_split(H2, OtherPos, test_size=0.3, random_state=42)

reg_h1 = Ridge(alpha=1.0).fit(H1_tr, SP_tr)
reg_h2 = Ridge(alpha=1.0).fit(H2_tr, OP_tr)
r2_h1  = r2_score(SP_te, reg_h1.predict(H1_te))
r2_h2  = r2_score(OP_te, reg_h2.predict(H2_te))

print(f"  h¹ → A-1 位置 R² = {r2_h1:.4f}  {'[OK]' if r2_h1 > 0.8 else '[FAIL]'}")
print(f"  h² → A-2 位置 R² = {r2_h2:.4f}  {'[OK]' if r2_h2 > 0.8 else '[FAIL]'}")

# =========================================================================
# 描画
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f'Iizuka model (exp1_l1) as Phase 1 — Position Encoding Verification\n'
    f'R²(h¹→A-1)={r2_h1:.3f}  R²(h²→A-2)={r2_h2:.3f}  '
    f'({"PASS" if r2_h1>0.8 and r2_h2>0.8 else "FAIL"})',
    fontsize=11
)

ax = axes[0]
ax.scatter(H1_2d[:, 0], H1_2d[:, 1], c=colors_self, s=2, alpha=0.4)
ax.set_title(f'h¹  (A-1 position)\nPC1={var1[0]:.2f}, PC2={var1[1]:.2f}, R²={r2_h1:.3f}', fontsize=9)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True, alpha=0.2)

ax = axes[1]
ax.scatter(H2_2d[:, 0], H2_2d[:, 1], c=colors_other, s=2, alpha=0.4)
ax.set_title(f'h²  (A-2 position)\nPC1={var2[0]:.2f}, PC2={var2[1]:.2f}, R²={r2_h2:.3f}', fontsize=9)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True, alpha=0.2)

ax = axes[2]
gx, gy = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
ax.imshow(np.stack([gy, gx+gy-2*gx*gy, (1-gx)*(1-gy)], axis=2),
          origin='lower', extent=[-WORLD_HALF, WORLD_HALF, -WORLD_HALF, WORLD_HALF])
ax.set_title('Color reference\nBL=Blue, BR=Green, TL=Yellow, TR=Red', fontsize=9)
ax.set_xlabel('x'); ax.set_ylabel('y')

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

print("\n=== 検証サマリ ===")
print(f"  R²(h¹→A-1) = {r2_h1:.4f}  {'[OK]' if r2_h1 > 0.8 else '[FAIL]'}")
print(f"  R²(h²→A-2) = {r2_h2:.4f}  {'[OK]' if r2_h2 > 0.8 else '[FAIL]'}")
iizuka_ok = r2_h1 > 0.8 and r2_h2 > 0.8
print(f"\n  飯塚モデル = Phase 1 として使える: {'YES' if iizuka_ok else 'NO'}")
if iizuka_ok:
    print("  → exp1_l1 checkpoint を frozen にして Phase 2 (MG 訓練) へ進む")
else:
    print("  → 評価戦略の見直しが必要")
