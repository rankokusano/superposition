"""
visualize_pca_spatial_mg6.py

mg6 モデルの Shared Module 隠れ状態を PCA で可視化。
step2/mg5/visualize_pca_spatial_mg5.py の mg6 対応版。

mg5 との変更点:
    - SuperpositionNetworkWithMG2 → SuperpositionNetworkWithMG3
    - MODEL_FILE: model_mg5.pth → model_mg6.pth
    - SAVE_PATH: pca_spatial_mg5.png → pca_spatial_mg6.png
    - RUN_DIR のデフォルト: step2/results → step3/results

確認したいこと:
    mg5 の問題: h² → A-1 R² ≈ 0.879 ≈ h² → A-2 R² ≈ 0.862  (自他分離なし)
    mg6 の期待: h² → A-2 R² が高く、h² → A-1 R² が低くなっているか

実行コマンド（コンテナ内）:
    cd /work
    Xvfb :99 -screen 0 1024x768x24 & sleep 1 && \\
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/mg6/visualize_pca_spatial_mg6.py \\
        --run mg6_YYYYMMDD_HHMMSS \\
    2>&1 | tee my_research/step3/results/mg6_YYYYMMDD_HHMMSS/pca_spatial_log.txt

出力:
    {run_dir}/pca_spatial_mg6.png
    標準出力に R² スコアを表示
"""

import sys, os, argparse

_HERE       = os.path.dirname(os.path.abspath(__file__))
STEP3_DIR   = os.path.abspath(os.path.join(_HERE, '..'))
STEP2_DIR   = os.path.abspath(os.path.join(_HERE, '../../step2'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP3_DIR, STEP2_DIR, MY_RESEARCH, PROJ_ROOT]:
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
from model_mg3 import SuperpositionNetworkWithMG3

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()

RUN_DIR    = (args.run if os.path.isabs(args.run)
              else os.path.join(STEP3_DIR, 'results', args.run))
MODEL_FILE = os.path.join(RUN_DIR, 'model_mg6.pth')
SAVE_PATH  = os.path.join(RUN_DIR, 'pca_spatial_mg6.png')

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
ve_input  = ckpt.get('ve_input',  'ov_enc+om_generated')
print(f"  episodes={ckpt.get('episodes','?')}  "
      f"final_loss={ckpt.get('final_loss', float('nan')):.5f}  "
      f"ve_input={ve_input}")

model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithMG3(model_config, q_dim=1, mg_hidden=mg_hidden).to(DEVICE)
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
def pos_to_color(x, y):
    u = np.clip((x + ARENA) / (2 * ARENA), 0, 1)
    v = np.clip((y + ARENA) / (2 * ARENA), 0, 1)
    c_bl = np.array([0.0, 0.7, 0.0])
    c_br = np.array([0.0, 0.0, 1.0])
    c_tl = np.array([1.0, 0.0, 0.0])
    c_tr = np.array([0.0, 1.0, 1.0])
    return np.clip(
        (1-u)*(1-v)*c_bl + u*(1-v)*c_br +
        (1-u)*v   *c_tl + u*v   *c_tr, 0, 1)

# =========================================================================
print(f"\n隠れ状態収集中 ({N_EPISODES} ep x {MAX_STEPS} steps)...")

all_h1, all_h2 = [], []
all_pos_a1, all_pos_a2 = [], []

for ep in range(N_EPISODES):
    env.reset()
    net.init_state(1)

    v_raw, _, _, self_pos, other_pos = env.step()
    actor_h = critic_h = None

    for step in range(MAX_STEPS):
        v_raw_t = torch.FloatTensor(
            v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        v_t = torch.FloatTensor(
            scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action, _, actor_h = actor.sample(v_raw_t, actor_h)
            q_raw,  critic_h   = critic(v_raw_t, action, critic_h)
            q_self             = q_raw / Q_SCALE

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

H1   = np.array(all_h1)
H2   = np.array(all_h2)
P_a1 = np.array(all_pos_a1)
P_a2 = np.array(all_pos_a2)
C_a1 = np.array([pos_to_color(x, y) for x, y in P_a1])
C_a2 = np.array([pos_to_color(x, y) for x, y in P_a2])
print(f"収集完了: {len(H1)} ステップ")

# =========================================================================
pca  = PCA(n_components=2)
pca.fit(H1)
H1_2d = pca.transform(H1)
H2_2d = pca.transform(H2)
var   = pca.explained_variance_ratio_
print(f"PCA 寄与率: PC1={var[0]:.3f}, PC2={var[1]:.3f} (合計={var.sum():.3f})")

# =========================================================================
print("\n=== 線形回帰: 隠れ状態 -> エージェント位置 (R^2) ===")
print("  mg5 結果参照: h1->A-1: 0.986, h2->A-2: 0.862, h2->A-1: 0.879 (問題)")
print("  mg6 期待: h2->A-2 が高く、h2->A-1 が低くなること")
results = {}
for name, H, P in [
    ("h1 -> A-1 (自己,  期待: 高)", H1, P_a1),
    ("h2 -> A-2 (他者,  期待: 高)", H2, P_a2),
    ("h1 -> A-2 (交差,  期待: 低)", H1, P_a2),
    ("h2 -> A-1 (交差,  期待: 低)", H2, P_a1),
]:
    rx = cross_val_score(Ridge(), H, P[:, 0], cv=5, scoring='r2').mean()
    ry = cross_val_score(Ridge(), H, P[:, 1], cv=5, scoring='r2').mean()
    print(f"  {name}:  R2_x={rx:.3f}  R2_y={ry:.3f}")
    results[name] = (rx, ry)

# 自他分離の改善度を数値で表示
h2_a2 = np.mean(results["h2 -> A-2 (他者,  期待: 高)"])
h2_a1 = np.mean(results["h2 -> A-1 (交差,  期待: 低)"])
print(f"\n  [自他分離度] h2->A-2: {h2_a2:.3f}  h2->A-1: {h2_a1:.3f}  差: {h2_a2 - h2_a1:+.3f}")
print(f"  mg5 の差: 0.862 - 0.879 = -0.017  (A-1 の方が高かった)")

# =========================================================================
fig = plt.figure(figsize=(16, 6))
fig.suptitle(
    f'PCA of Shared Module hidden states  [mg6: {os.path.basename(RUN_DIR)}]\n'
    f'VE input: (ov_enc, om_generated)  [mg5 は h2_t-1]\n'
    f'PC1={var[0]:.2f}, PC2={var[1]:.2f}  '
    f'({N_EPISODES} ep x {MAX_STEPS} steps)',
    fontsize=11
)

ax1 = fig.add_subplot(1, 3, 1)
ax1.scatter(H1_2d[:, 0], H1_2d[:, 1], c=C_a1, s=4, alpha=0.5)
ax1.set_title('Process-1  h1\n(A-1 pos colored)', fontsize=10)
ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
ax1.grid(True, alpha=0.2)

ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(H2_2d[:, 0], H2_2d[:, 1], c=C_a2, s=4, alpha=0.5)
ax2.set_title('Process-2  h2\n(A-2 pos colored)', fontsize=10)
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
ax3.set_title('Color map\n(same as paper Fig.4c)', fontsize=10)
ax3.set_xlabel('X'); ax3.set_ylabel('Y')
ax3.legend(fontsize=8, loc='center')

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

# =========================================================================
print("\n=== 空間構造チェック（PCA 象限ごとの平均色の差） ===")
for label, H_2d, C in [('Process-1 h1', H1_2d, C_a1),
                         ('Process-2 h2', H2_2d, C_a2)]:
    q1 = (H_2d[:,0] > np.median(H_2d[:,0])) & (H_2d[:,1] > np.median(H_2d[:,1]))
    q3 = (H_2d[:,0] < np.median(H_2d[:,0])) & (H_2d[:,1] < np.median(H_2d[:,1]))
    if q1.sum() > 0 and q3.sum() > 0:
        diff = np.linalg.norm(C[q1].mean(0) - C[q3].mean(0))
        print(f"  {label}: 対角象限の色差 = {diff:.3f}")
