"""
visualize_phase2.py

Phase 2 検証スクリプト。

【検証項目】
    1. MG 相関: MG が出力する om と実際の A-2 モーション の Pearson r
       → 目標: r ≥ 0.87 (Noguchi et al. の基準)
    2. h² 選好分離: h² を PCA して 4 選好で色分け → クラスタ分離
    3. h² 選好デコード: h² からの選好分類精度

【評価方法】
    4 選好 × 50 ep で固定 LandmarkFollowerAgent を使用。
    各エピソードの全ステップの om / h² を収集。

実行コマンド（コンテナ内）:
    cd /work
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/phase2/visualize_phase2.py \\
        --model my_research/step3/results/phase2_20260615_182451/model_phase2.pth \\
    2>&1 | tee my_research/step3/results/visualize_phase2.log
"""

import sys, os, argparse

_HERE       = os.path.dirname(os.path.abspath(__file__))
STEP3_DIR   = os.path.abspath(os.path.join(_HERE, '..'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP3_DIR, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from util import scale_vision
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_multi_pref import LandmarkFollowerAgent, LANDMARK_NAMES
from model.model import SuperpositionNetworkMotionGenerationFeaturePrediction

# =========================================================================
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_EP_EACH  = 50     # 各選好につきエピソード数
MAX_STEPS  = 80
WORLD_HALF = 9.5

IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkMotionGenerationFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

PREF_COLORS = {'Red': '#e74c3c', 'Green': '#2ecc71', 'Blue': '#3498db', 'Cyan': '#1abc9c'}

# =========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Phase 2 model path (.pth)')
args = parser.parse_args()

MODEL_PATH = args.model
RUN_DIR    = os.path.dirname(MODEL_PATH)
SAVE_PATH  = os.path.join(RUN_DIR, 'visualize_phase2.png')

# =========================================================================
# モデル・環境のロード
# =========================================================================
print(f"モデル読み込み: {MODEL_PATH}")
model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkMotionGenerationFeaturePrediction(model_config).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = ckpt.get('net_state_dict', ckpt)
net.load_state_dict(state_dict)
net.eval()
for p in net.parameters():
    p.requires_grad = False
print("  読み込み完了")

actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()

critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
critic.eval()

env_config = load_config(ENV_CONFIG_PATH)
env = creator.create_environment(env_config.environment)
env.init(); env.off_display()

# =========================================================================
# データ収集
# =========================================================================
all_om_pred   = []   # MG 予測モーション (N, 2)
all_om_true   = []   # 実際の A-2 モーション (N, 2)
all_h2        = []   # h² 隠れ状態 (N, 128)
all_pref      = []   # 選好ラベル (N,) integer
all_pref_name = []   # 選好名 (N,) string
all_other_pos = []   # A-2 実際位置 (N, 2)

print(f"\nデータ収集中 ({len(LANDMARK_NAMES)} 選好 × {N_EP_EACH} ep × {MAX_STEPS} steps)...")

for pref_idx, pref_name in enumerate(LANDMARK_NAMES):
    agent2 = LandmarkFollowerAgent(pref_name)
    ep_collected = 0
    while ep_collected < N_EP_EACH:
        env.reset()
        net.init_state(1)
        actor_h = critic_h = None
        v_raw, _, _, self_pos, other_pos = env.step()

        for step in range(MAX_STEPS):
            v_raw_t = torch.FloatTensor(
                v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            v_t = torch.FloatTensor(
                scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action, _, actor_h = actor.sample(v_raw_t, actor_h)
                _, critic_h = critic(v_raw_t, action, critic_h)

                om_true_np = agent2.get_action(other_pos)

                p_mask = 0.0 if step == 0 else 1.0
                pred = net(
                    {'self_vision': v_t, 'self_motion': action},
                    p_mask_vision_self=p_mask,
                    p_mask_vision_other=p_mask,
                )

                # MG 出力
                om_pred_np = pred['other_motion'].squeeze(0).cpu().numpy()

                # h²
                state = net.superposition_module.state
                h2 = state['other'].hidden.squeeze(0).cpu().numpy()

            all_om_pred.append(om_pred_np)
            all_om_true.append(om_true_np)
            all_h2.append(h2)
            all_pref.append(pref_idx)
            all_pref_name.append(pref_name)
            all_other_pos.append(other_pos.copy())

            a_np      = action.squeeze(0).cpu().numpy()
            self_pos  = np.clip(self_pos  + a_np,      -WORLD_HALF, WORLD_HALF)
            other_pos = np.clip(other_pos + om_true_np, -WORLD_HALF, WORLD_HALF)
            env.self_agent.p  = self_pos
            env.other_agent.p = other_pos
            v_raw, _, _, _, _ = env.step()

        ep_collected += 1

    print(f"  {pref_name}: {N_EP_EACH} ep 完了")

om_pred   = np.array(all_om_pred)    # (N, 2)
om_true   = np.array(all_om_true)    # (N, 2)
H2        = np.array(all_h2)         # (N, 128)
pref_ids  = np.array(all_pref)       # (N,)
OtherPos  = np.array(all_other_pos)  # (N, 2)
N_total   = len(H2)
print(f"\n合計サンプル数: {N_total}")

# =========================================================================
# 検証 1: MG 相関
# =========================================================================
print("\n=== 検証 1: MG 相関 (Pearson r) ===")

r_x, p_x = pearsonr(om_pred[:, 0], om_true[:, 0])
r_y, p_y = pearsonr(om_pred[:, 1], om_true[:, 1])
r_mean   = (r_x + r_y) / 2

# ベクトル全体の相関 (各サンプルの内積の相関)
# Noguchi et al. の方法に近い: 速度ベクトルの各次元の r の平均
print(f"  r_x (x 方向): {r_x:.4f}  (p={p_x:.2e})")
print(f"  r_y (y 方向): {r_y:.4f}  (p={p_y:.2e})")
print(f"  r_mean      : {r_mean:.4f}  {'[OK >= 0.87]' if r_mean >= 0.87 else '[FAIL < 0.87]'}")

# =========================================================================
# 検証 2: h² PCA 選好分離
# =========================================================================
print("\n=== 検証 2: h² PCA 選好分離 ===")

pca = PCA(n_components=2)
H2_2d = pca.fit_transform(H2)
var = pca.explained_variance_ratio_
print(f"  h² PCA 寄与率: PC1={var[0]:.3f}, PC2={var[1]:.3f}, 合計={var.sum():.3f}")

# 選好間 / 全分散 比
pref_centroids = [H2_2d[pref_ids == i].mean(axis=0) for i in range(4)]
grand_mean = H2_2d.mean(axis=0)
between_var = np.mean([np.sum((c - grand_mean)**2) for c in pref_centroids])
total_var   = np.var(H2_2d)
sep_score   = between_var / (total_var + 1e-8)
print(f"  選好分離スコア (between/total): {sep_score:.4f}")

# =========================================================================
# 検証 3: h² 選好分類精度
# =========================================================================
print("\n=== 検証 3: h² 線形分類精度 ===")

X_tr, X_te, y_tr, y_te = train_test_split(H2, pref_ids, test_size=0.3, random_state=42)
clf = LogisticRegression(max_iter=1000, C=1.0).fit(X_tr, y_tr)
acc = accuracy_score(y_te, clf.predict(X_te))
print(f"  Logistic Regression 精度: {acc:.4f}  (chance=0.25)")

# =========================================================================
# 描画
# =========================================================================
fig = plt.figure(figsize=(18, 10))
fig.suptitle(
    f'Phase 2 Verification — MG Correlation & h² Preference Separation\n'
    f'r_x={r_x:.3f}  r_y={r_y:.3f}  r_mean={r_mean:.3f}  '
    f'clf_acc={acc:.3f}  sep={sep_score:.3f}',
    fontsize=11
)

gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

# --- (0,0) MG 相関 x 方向 ---
ax = fig.add_subplot(gs[0, 0])
lim = max(np.abs(om_true[:, 0]).max(), np.abs(om_pred[:, 0]).max()) * 1.1
ax.scatter(om_true[:, 0], om_pred[:, 0],
           c=[PREF_COLORS[LANDMARK_NAMES[i]] for i in pref_ids],
           s=1, alpha=0.3)
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('Actual vx'); ax.set_ylabel('MG predicted vx')
ax.set_title(f'MG corr x: r={r_x:.3f}', fontsize=9)
ax.set_aspect('equal')

# --- (0,1) MG 相関 y 方向 ---
ax = fig.add_subplot(gs[0, 1])
lim = max(np.abs(om_true[:, 1]).max(), np.abs(om_pred[:, 1]).max()) * 1.1
ax.scatter(om_true[:, 1], om_pred[:, 1],
           c=[PREF_COLORS[LANDMARK_NAMES[i]] for i in pref_ids],
           s=1, alpha=0.3)
ax.plot([-lim, lim], [-lim, lim], 'k--', lw=0.8, alpha=0.6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('Actual vy'); ax.set_ylabel('MG predicted vy')
ax.set_title(f'MG corr y: r={r_y:.3f}', fontsize=9)
ax.set_aspect('equal')

# --- (0,2) h² PCA 選好分布 ---
ax = fig.add_subplot(gs[0, 2])
for i, pref_name in enumerate(LANDMARK_NAMES):
    mask = pref_ids == i
    ax.scatter(H2_2d[mask, 0], H2_2d[mask, 1],
               c=PREF_COLORS[pref_name], s=2, alpha=0.3, label=pref_name)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title(f'h² PCA (sep={sep_score:.3f}, acc={acc:.3f})', fontsize=9)
ax.legend(fontsize=7, markerscale=3)
ax.grid(True, alpha=0.2)

# --- (0,3) MG 出力分布 (各選好の om 分布) ---
ax = fig.add_subplot(gs[0, 3])
for i, pref_name in enumerate(LANDMARK_NAMES):
    mask = pref_ids == i
    c = PREF_COLORS[pref_name]
    ax.scatter(om_pred[mask, 0], om_pred[mask, 1],
               c=c, s=2, alpha=0.3, label=pref_name)
    # 重心
    m = om_pred[mask].mean(axis=0)
    ax.scatter(*m, c=c, s=80, marker='*', edgecolors='k', linewidths=0.5, zorder=5)
ax.set_xlabel('om_x (predicted)'); ax.set_ylabel('om_y (predicted)')
ax.set_title('MG output distribution per preference', fontsize=9)
ax.legend(fontsize=7, markerscale=3)
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
ax.axvline(0, color='k', lw=0.5, alpha=0.3)
ax.grid(True, alpha=0.2)

# --- (1,0) 実際の A-2 モーション分布 ---
ax = fig.add_subplot(gs[1, 0])
for i, pref_name in enumerate(LANDMARK_NAMES):
    mask = pref_ids == i
    c = PREF_COLORS[pref_name]
    ax.scatter(om_true[mask, 0], om_true[mask, 1],
               c=c, s=2, alpha=0.3, label=pref_name)
    m = om_true[mask].mean(axis=0)
    ax.scatter(*m, c=c, s=80, marker='*', edgecolors='k', linewidths=0.5, zorder=5)
ax.set_xlabel('actual vx'); ax.set_ylabel('actual vy')
ax.set_title('Actual A-2 motion distribution', fontsize=9)
ax.legend(fontsize=7, markerscale=3)
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
ax.axvline(0, color='k', lw=0.5, alpha=0.3)
ax.grid(True, alpha=0.2)

# --- (1,1) h² PCA 時系列軌跡 (4 選好×1 ep) ---
ax = fig.add_subplot(gs[1, 1])
steps_per_pref = N_EP_EACH * MAX_STEPS
for i, pref_name in enumerate(LANDMARK_NAMES):
    start = i * steps_per_pref
    traj = H2_2d[start:start + MAX_STEPS]
    ax.plot(traj[:, 0], traj[:, 1],
            c=PREF_COLORS[pref_name], lw=1.0, alpha=0.8, label=pref_name)
    ax.scatter(traj[0, 0], traj[0, 1], c=PREF_COLORS[pref_name],
               s=40, marker='o', zorder=5)
    ax.scatter(traj[-1, 0], traj[-1, 1], c=PREF_COLORS[pref_name],
               s=40, marker='X', zorder=5)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('h² trajectory (1 ep per pref, o=start, X=end)', fontsize=9)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)

# --- (1,2) h² PCA 位置マップ (A-2 位置で色付け) ---
ax = fig.add_subplot(gs[1, 2])
nx = np.clip((OtherPos[:, 0] + WORLD_HALF) / (2 * WORLD_HALF), 0, 1)
ny = np.clip((OtherPos[:, 1] + WORLD_HALF) / (2 * WORLD_HALF), 0, 1)
colors_pos = np.stack([ny, nx + ny - 2*nx*ny, (1-nx)*(1-ny)], axis=1)
ax.scatter(H2_2d[:, 0], H2_2d[:, 1], c=colors_pos, s=2, alpha=0.3)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('h² PCA colored by A-2 position', fontsize=9)
ax.grid(True, alpha=0.2)

# --- (1,3) サマリテキスト ---
ax = fig.add_subplot(gs[1, 3])
ax.axis('off')
summary_lines = [
    "=== Phase 2 Summary ===",
    "",
    f"MG Pearson r_x  : {r_x:.4f}",
    f"MG Pearson r_y  : {r_y:.4f}",
    f"MG Pearson mean : {r_mean:.4f}",
    f"  target ≥ 0.87 : {'OK' if r_mean >= 0.87 else 'FAIL'}",
    "",
    f"h² sep score    : {sep_score:.4f}",
    f"h² clf acc      : {acc:.4f}  (chance=0.25)",
    f"  > 0.5: {'OK' if acc > 0.5 else 'FAIL'}",
    "",
    f"PCA var PC1+PC2 : {var.sum():.3f}",
    "",
    f"N samples       : {N_total}",
    f"N pref x N_ep   : {len(LANDMARK_NAMES)} x {N_EP_EACH}",
    f"Max steps       : {MAX_STEPS}",
]
ax.text(0.05, 0.95, '\n'.join(summary_lines),
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

print("\n=== Phase 2 検証サマリ ===")
print(f"  MG r_x  = {r_x:.4f}")
print(f"  MG r_y  = {r_y:.4f}")
print(f"  MG mean = {r_mean:.4f}  {'[OK]' if r_mean >= 0.87 else '[FAIL]'}")
print(f"  h² clf acc = {acc:.4f}  {'[OK]' if acc > 0.5 else '[FAIL]'}")
print(f"  sep score  = {sep_score:.4f}")
