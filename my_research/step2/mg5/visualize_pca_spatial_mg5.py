"""
visualize_pca_spatial_mg5.py

mg5 モデルの Shared Module 隠れ状態を PCA で可視化。
Noguchi et al. (2022) Fig. 4c の再現に相当。

確認したいこと:
  h¹ が A-1 の位置に対応する空間構造を持つか
  h² が A-2 の位置に対応する空間構造を持つか
  → どちらも成立すれば「社会的場所細胞」的な表現が得られている

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_pca_spatial_mg5.py \\
        --run mg5_YYYYMMDD_HHMMSS

出力:
    {run_dir}/pca_spatial_mg5.png
    標準出力に R² スコアを表示
"""

import sys, os, argparse

_HERE       = os.path.dirname(os.path.abspath(__file__))
STEP2_DIR   = os.path.abspath(os.path.join(_HERE, '..'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP2_DIR, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from util import scale_vision
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_green import GreenFollowerAgent
from model_mg2 import SuperpositionNetworkWithMG2

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()

RUN_DIR    = (args.run if os.path.isabs(args.run)
              else os.path.join(STEP2_DIR, 'results', args.run))
MODEL_FILE = os.path.join(RUN_DIR, 'model_mg5.pth')
SAVE_PATH  = os.path.join(RUN_DIR, 'pca_spatial_mg5.png')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

N_EPISODES = 50
MAX_STEPS  = 80
ARENA      = 9.0

# =========================================================================
print(f"run: {RUN_DIR}")
if not os.path.exists(MODEL_FILE):
    print(f"[ERROR] {MODEL_FILE} が見つかりません"); sys.exit(1)

ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
Q_SCALE   = ckpt.get('q_scale',   50.0)
mg_hidden = ckpt.get('mg_hidden', 64)
print(f"  episodes={ckpt.get('episodes','?')}  "
      f"final_loss={ckpt.get('final_loss', float('nan')):.5f}  "
      f"encoder_frozen={ckpt.get('encoder_frozen', '?')}")

model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithMG2(model_config, q_dim=1, mg_hidden=mg_hidden).to(DEVICE)
net.load_state_dict(ckpt['net_state_dict'])
net.eval()
for p in net.parameters(): p.requires_grad = False

actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()

critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
critic.eval()

env_config = load_config(ENV_CONFIG_PATH)
env = creator.create_environment(env_config.environment)
env.init(); env.off_display()

green_follower = GreenFollowerAgent()

# =========================================================================
# 位置 → 色（論文 Fig.4c と同じカラーマップ）
# =========================================================================
def pos_to_color(x, y):
    u = np.clip((x + ARENA) / (2 * ARENA), 0, 1)
    v = np.clip((y + ARENA) / (2 * ARENA), 0, 1)
    c_bl = np.array([0.0, 0.7, 0.0])  # Green  (-9,-9)
    c_br = np.array([0.0, 0.0, 1.0])  # Blue   (+9,-9)
    c_tl = np.array([1.0, 0.0, 0.0])  # Red    (-9,+9)
    c_tr = np.array([0.0, 1.0, 1.0])  # Cyan   (+9,+9)
    return np.clip(
        (1-u)*(1-v)*c_bl + u*(1-v)*c_br +
        (1-u)*v   *c_tl + u*v   *c_tr, 0, 1)

# =========================================================================
# 隠れ状態の収集
# =========================================================================
print(f"\n隠れ状態収集中 ({N_EPISODES} ep × {MAX_STEPS} steps)...")

all_h1, all_h2 = [], []
all_pos_a1, all_pos_a2 = [], []

for ep in range(N_EPISODES):
    env.reset()
    net.init_state(1)

    v_raw, _, _, self_pos, other_pos = env.step()
    actor_h = critic_h = None

    for step in range(MAX_STEPS):
        # Actor/Critic: [0,1] のまま（学習時と同じドメイン）
        v_raw_t = torch.FloatTensor(
            v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        # encoder: [-1,1]（exp1 学習時と同じドメイン）
        v_t = torch.FloatTensor(
            scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action, _, actor_h  = actor.sample(v_raw_t, actor_h)
            q_raw,  critic_h    = critic(v_raw_t, action, critic_h)
            q_self              = q_raw / Q_SCALE

            om_np = green_follower.get_action(other_pos)

            _, h1, h2 = net(
                {'self_vision': v_t, 'self_motion': action},
                q_self, 0.0, 0.0
            )
            net.detach_state()

        all_h1.append(h1.squeeze(0).cpu().numpy())
        all_h2.append(h2.squeeze(0).cpu().numpy())
        all_pos_a1.append(self_pos.copy())
        all_pos_a2.append(other_pos.copy())

        a_np      = action.squeeze(0).cpu().numpy()
        new_self  = np.clip(self_pos  + a_np,  -9.5, 9.5)
        new_other = np.clip(other_pos + om_np, -9.5, 9.5)
        env.self_agent.p  = new_self
        env.other_agent.p = new_other
        v_raw, _, _, _, _ = env.step()
        self_pos  = new_self
        other_pos = new_other

    if (ep + 1) % 10 == 0:
        print(f"  {ep+1}/{N_EPISODES} done")

H1   = np.array(all_h1)     # (T, 128)
H2   = np.array(all_h2)
P_a1 = np.array(all_pos_a1) # (T, 2)
P_a2 = np.array(all_pos_a2)
C_a1 = np.array([pos_to_color(x, y) for x, y in P_a1])
C_a2 = np.array([pos_to_color(x, y) for x, y in P_a2])
print(f"収集完了: {len(H1)} ステップ")

# =========================================================================
# PCA: H1 で fit し、H1/H2 両方を同じ軸に射影（論文準拠）
# =========================================================================
pca  = PCA(n_components=2)
pca.fit(H1)
H1_2d = pca.transform(H1)
H2_2d = pca.transform(H2)
var   = pca.explained_variance_ratio_
print(f"PCA 寄与率: PC1={var[0]:.3f}, PC2={var[1]:.3f} (合計={var.sum():.3f})")

# =========================================================================
# 線形回帰: 隠れ状態 → 位置の R²（論文 Fig.4d 相当）
# =========================================================================
print("\n=== 線形回帰: 隠れ状態 → エージェント位置 (R²) ===")
print("  論文の期待: h¹→A-1 高, h²→A-2 高, 交差項は低")
for name, H, P in [
    ("h¹ → A-1 位置 (自己,  期待: 高)", H1, P_a1),
    ("h² → A-2 位置 (他者,  期待: 高)", H2, P_a2),
    ("h¹ → A-2 位置 (交差,  期待: 低)", H1, P_a2),
    ("h² → A-1 位置 (交差,  期待: 低)", H2, P_a1),
]:
    rx = cross_val_score(Ridge(), H, P[:, 0], cv=5, scoring='r2').mean()
    ry = cross_val_score(Ridge(), H, P[:, 1], cv=5, scoring='r2').mean()
    print(f"  {name}:  R²_x={rx:.3f}  R²_y={ry:.3f}")

# =========================================================================
# 可視化: 3 パネル
# =========================================================================
fig = plt.figure(figsize=(16, 6))
fig.suptitle(
    f'PCA of Shared Module hidden states  [mg5: {os.path.basename(RUN_DIR)}]\n'
    f'h¹ → A-1 の位置構造、h² → A-2 の位置構造\n'
    f'PC1={var[0]:.2f}, PC2={var[1]:.2f}  '
    f'({N_EPISODES} ep × {MAX_STEPS} steps)',
    fontsize=11
)

ax1 = fig.add_subplot(1, 3, 1)
ax1.scatter(H1_2d[:, 0], H1_2d[:, 1], c=C_a1, s=4, alpha=0.5)
ax1.set_title('Process-1  h¹\n（A-1 の位置で着色）', fontsize=10)
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
ax1.grid(True, alpha=0.2)

ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(H2_2d[:, 0], H2_2d[:, 1], c=C_a2, s=4, alpha=0.5)
ax2.set_title('Process-2  h²\n（A-2 の位置で着色）', fontsize=10)
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
ax2.grid(True, alpha=0.2)

ax3 = fig.add_subplot(1, 3, 3)
ls = 200
gx = np.linspace(-ARENA, ARENA, ls)
gy = np.linspace(-ARENA, ARENA, ls)
GX, GY = np.meshgrid(gx, gy)
legend_img = np.array([[pos_to_color(GX[i,j], GY[i,j])
                         for j in range(ls)] for i in range(ls)])
ax3.imshow(legend_img, extent=[-ARENA, ARENA, -ARENA, ARENA],
           origin='lower', aspect='equal')
for name, lx, ly, lc in [('Red(-9,+9)',  -9, 9,  'red'),
                           ('Green(-9,-9)',-9, -9, 'limegreen'),
                           ('Blue(+9,-9)', 9,  -9, 'blue'),
                           ('Cyan(+9,+9)', 9,  9,  'cyan')]:
    ax3.plot(lx, ly, '*', color=lc, markersize=14,
             markeredgecolor='black', markeredgewidth=0.6, label=name)
ax3.set_title('位置→色 凡例\n（論文 Fig.4c と同じ）', fontsize=10)
ax3.set_xlabel('X'); ax3.set_ylabel('Y')
ax3.legend(fontsize=8, loc='center')

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

# =========================================================================
# 追加チェック: PCA 空間の四隅で色が分かれているか
# =========================================================================
print("\n=== 空間構造チェック（PCA 象限ごとの平均色の差） ===")
print("  差が大きいほど位置に応じた構造あり（期待: どちらも高い値）")
for label, H_2d, C in [('Process-1 h¹', H1_2d, C_a1),
                         ('Process-2 h²', H2_2d, C_a2)]:
    q1 = (H_2d[:,0] > np.median(H_2d[:,0])) & (H_2d[:,1] > np.median(H_2d[:,1]))
    q3 = (H_2d[:,0] < np.median(H_2d[:,0])) & (H_2d[:,1] < np.median(H_2d[:,1]))
    if q1.sum() > 0 and q3.sum() > 0:
        diff = np.linalg.norm(C[q1].mean(0) - C[q3].mean(0))
        print(f"  {label}: 対角象限の色差 = {diff:.3f}")
