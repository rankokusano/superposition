"""
visualize_phase3.py

Phase 3: h² による選好推論の時間発展可視化 + 位置追跡 baseline 比較。

【可視化項目】
    1. 時刻ごとの選好分類精度 (accuracy over time) — 3 曲線を比較
       a) h²  (128次元): SM の内部状態から分類
       b) pos (2次元): A-2 の実際の (x,y) 座標から分類  ← baseline
       c) om_pred (2次元): MG が出力した運動ベクトルから分類  ← baseline

       解釈の枠組み:
         h² ≈ pos          → SM は現在位置を追跡しているだけ（trivial）
         h² > pos (早期 t) → SM が位置以上の情報（軌跡・移動方向）を符号化
         h² > om_pred      → LSTM 記憶が瞬時の MG 出力より多くの情報を統合

    2. h² アトラクター軌跡 (PCA, 複数エピソード)
    3. 選好ごとの PC1 の時間推移 (mean ± std)
    4. 混同行列

実行コマンド（コンテナ内）:
    cd /work
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/phase3/visualize_phase3.py \\
        --model my_research/step3/results/phase2_offline_20260616_052720/model_best.pth \\
    2>&1 | tee my_research/step3/results/phase2_offline_20260616_052720/phase3.log
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
import matplotlib.cm as cm
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from util import scale_vision
from rl_agent_sac import ActorLSTM
from agent2_multi_pref import LandmarkFollowerAgent, LANDMARK_NAMES
from model.model import SuperpositionNetworkMotionGenerationFeaturePrediction

# =========================================================================
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_EP_EACH  = 80      # 各選好のエピソード数（多いほど統計が安定）
MAX_STEPS  = 80
WORLD_HALF = 9.5

IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkMotionGenerationFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

PREF_COLORS = {
    'Red':   '#e74c3c',
    'Green': '#27ae60',
    'Blue':  '#2980b9',
    'Cyan':  '#16a085',
}
LANDMARKS_XY = {
    'Red':   (-9,  9),
    'Green': (-9, -9),
    'Blue':  ( 9, -9),
    'Cyan':  ( 9,  9),
}

# =========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
args   = parser.parse_args()

MODEL_PATH = args.model
RUN_DIR    = os.path.dirname(MODEL_PATH)
SAVE_PATH  = os.path.join(RUN_DIR, 'phase3.png')

# =========================================================================
# モデル・環境ロード
# =========================================================================
print(f"モデル: {MODEL_PATH}")
model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkMotionGenerationFeaturePrediction(model_config).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
net.load_state_dict(ckpt.get('net_state_dict', ckpt))
net.eval()
for p in net.parameters():
    p.requires_grad = False

actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()

env_config = load_config(ENV_CONFIG_PATH)
env = creator.create_environment(env_config.environment)
env.init(); env.off_display()

# =========================================================================
# データ収集: (N_EP_EACH × 4) エピソード × MAX_STEPS ステップ
# h2_traj[pref_idx][ep][step] = h² (128,)
# =========================================================================
print(f"\nデータ収集中 ({len(LANDMARK_NAMES)} 選好 × {N_EP_EACH} ep × {MAX_STEPS} steps)...")

# shape: (4, N_EP_EACH, MAX_STEPS, dim)
h2_traj      = np.zeros((4, N_EP_EACH, MAX_STEPS, 128), dtype=np.float32)
om_pred_traj = np.zeros((4, N_EP_EACH, MAX_STEPS, 2),   dtype=np.float32)
om_true_traj = np.zeros((4, N_EP_EACH, MAX_STEPS, 2),   dtype=np.float32)
pos_traj     = np.zeros((4, N_EP_EACH, MAX_STEPS, 2),   dtype=np.float32)  # A-2 位置 baseline

for pref_idx, pref_name in enumerate(LANDMARK_NAMES):
    agent2 = LandmarkFollowerAgent(pref_name)
    for ep in range(N_EP_EACH):
        env.reset()
        net.init_state(1)
        actor_h = None
        v_raw, _, _, self_pos, other_pos = env.step()

        for step in range(MAX_STEPS):
            v_raw_t = torch.FloatTensor(
                v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            v_t = torch.FloatTensor(
                scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action, _, actor_h = actor.sample(v_raw_t, actor_h)
                om_np = agent2.get_action(other_pos)

                p_mask = 0.0 if step == 0 else 1.0
                pred = net(
                    {'self_vision': v_t, 'self_motion': action},
                    p_mask_vision_self=p_mask,
                    p_mask_vision_other=p_mask,
                )
                h2 = net.superposition_module.state['other'].hidden.squeeze(0).cpu().numpy()
                om_pred_np = pred['other_motion'].squeeze(0).cpu().numpy()

            h2_traj[pref_idx, ep, step]      = h2
            om_pred_traj[pref_idx, ep, step] = om_pred_np
            om_true_traj[pref_idx, ep, step] = om_np
            pos_traj[pref_idx, ep, step]     = other_pos.copy()

            a_np      = action.squeeze(0).cpu().numpy()
            self_pos  = np.clip(self_pos  + a_np,  -WORLD_HALF, WORLD_HALF)
            other_pos = np.clip(other_pos + om_np, -WORLD_HALF, WORLD_HALF)
            env.self_agent.p  = self_pos
            env.other_agent.p = other_pos
            v_raw, _, _, _, _ = env.step()

    print(f"  {pref_name}: {N_EP_EACH} ep 完了")

# =========================================================================
# 解析 1: 時刻ごとの選好分類精度（h², pos baseline, om_pred baseline の比較）
# =========================================================================
print("\n=== 解析 1: 時刻ごとの選好分類精度 ===")

y_labels = np.repeat(np.arange(4), N_EP_EACH)
n_samples = len(y_labels)
rng = np.random.RandomState(42)
split_idx = rng.permutation(n_samples)
split = int(n_samples * 0.7)

acc_h2      = []
acc_pos     = []
acc_om_pred = []

for t in range(MAX_STEPS):
    def _fit_acc(X_all):
        X_tr = X_all[split_idx[:split]]
        X_te = X_all[split_idx[split:]]
        y_tr = y_labels[split_idx[:split]]
        y_te = y_labels[split_idx[split:]]
        clf = LogisticRegression(max_iter=500, C=1.0).fit(X_tr, y_tr)
        return accuracy_score(y_te, clf.predict(X_te))

    acc_h2.append(_fit_acc(h2_traj[:, :, t, :].reshape(-1, 128)))
    acc_pos.append(_fit_acc(pos_traj[:, :, t, :].reshape(-1, 2)))
    acc_om_pred.append(_fit_acc(om_pred_traj[:, :, t, :].reshape(-1, 2)))

acc_h2      = np.array(acc_h2)
acc_pos     = np.array(acc_pos)
acc_om_pred = np.array(acc_om_pred)

print(f"{'step':>6} | {'h²':>6} | {'pos':>6} | {'om_pred':>7} | h²-pos")
for t in [0, 5, 10, 20, 40, 60, 79]:
    diff = acc_h2[t] - acc_pos[t]
    print(f"{t:>6} | {acc_h2[t]:.4f} | {acc_pos[t]:.4f} | {acc_om_pred[t]:.4f}  | {diff:+.4f}")
print(f"  chance: 0.2500")
print(f"  h²  max: {acc_h2.max():.4f} at step {acc_h2.argmax()}")
print(f"  pos max: {acc_pos.max():.4f} at step {acc_pos.argmax()}")

# =========================================================================
# 解析 2: 全ステップ合算での混同行列
# =========================================================================
print("\n=== 解析 2: 混同行列 ===")

X_all = h2_traj.reshape(-1, 128)
y_all = np.repeat(np.arange(4), N_EP_EACH * MAX_STEPS)
n_all = len(X_all)
split = int(n_all * 0.7)
idx_all = np.random.RandomState(42).permutation(n_all)
clf_all = LogisticRegression(max_iter=500, C=1.0).fit(
    X_all[idx_all[:split]], y_all[idx_all[:split]])
y_pred_all = clf_all.predict(X_all[idx_all[split:]])
y_true_all = y_all[idx_all[split:]]
cm_mat = confusion_matrix(y_true_all, y_pred_all)
acc_all = accuracy_score(y_true_all, y_pred_all)
print(f"  全体精度: {acc_all:.4f}  (chance=0.25)")
print("  混同行列 (行=true, 列=pred):")
for i, name in enumerate(LANDMARK_NAMES):
    print(f"    {name:5s}: {cm_mat[i]}")

# =========================================================================
# 解析 3: PCA (全エピソード)
# =========================================================================
pca = PCA(n_components=2)
X_all_2d = pca.fit_transform(X_all)
h2_traj_2d = X_all_2d.reshape(4, N_EP_EACH, MAX_STEPS, 2)
print(f"\n  PCA 寄与率: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

# =========================================================================
# 解析 4: MG 相関
# =========================================================================
om_pred_flat = om_pred_traj.reshape(-1, 2)
om_true_flat = om_true_traj.reshape(-1, 2)
r_x = pearsonr(om_pred_flat[:, 0], om_true_flat[:, 0])[0]
r_y = pearsonr(om_pred_flat[:, 1], om_true_flat[:, 1])[0]
print(f"\n  MG r_x={r_x:.4f}  r_y={r_y:.4f}  mean={((r_x+r_y)/2):.4f}")

# =========================================================================
# 描画
# =========================================================================
fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    f'Phase 3: Preference Inference via h²  '
    f'[MG r={((r_x+r_y)/2):.3f}, acc(all)={acc_all:.3f}, chance=0.25]',
    fontsize=13, fontweight='bold'
)
gs = fig.add_gridspec(3, 4, hspace=0.40, wspace=0.35)

# --- (0,0-1) 時刻ごとの選好分類精度（3 曲線比較） ---
ax = fig.add_subplot(gs[0, :2])
ts = range(MAX_STEPS)
ax.plot(ts, acc_h2,      color='steelblue', lw=2.2, label=f'h² (128-d)')
ax.plot(ts, acc_pos,     color='darkorange', lw=1.8, ls='--', label='pos baseline (x,y of A-2)')
ax.plot(ts, acc_om_pred, color='mediumseagreen', lw=1.6, ls=':', label='MG output (om_pred, 2-d)')
ax.axhline(0.25, color='gray', lw=1.2, ls='-', alpha=0.6, label='chance (0.25)')
ax.fill_between(ts, acc_pos, acc_h2,
                where=acc_h2 > acc_pos, alpha=0.12, color='steelblue',
                label='h² advantage over pos')
ax.set_xlabel('Observation step t', fontsize=10)
ax.set_ylabel('Preference classification accuracy', fontsize=10)
ax.set_title('Preference decoding accuracy over time\n'
             'h² vs position baseline vs MG output', fontsize=9)
ax.set_ylim(0.0, 1.0)
ax.set_xlim(0, MAX_STEPS - 1)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.25)

# --- (0,2-3) PC1/PC2 の時間推移 (mean ± std per pref) ---
ax = fig.add_subplot(gs[0, 2:])
ts = np.arange(MAX_STEPS)
for pref_idx, pref_name in enumerate(LANDMARK_NAMES):
    traj = h2_traj_2d[pref_idx]   # (N_EP, MAX_STEPS, 2)
    pc1_mean = traj[:, :, 0].mean(axis=0)
    pc1_std  = traj[:, :, 0].std(axis=0)
    c = PREF_COLORS[pref_name]
    ax.plot(ts, pc1_mean, color=c, lw=1.8, label=pref_name)
    ax.fill_between(ts, pc1_mean - pc1_std, pc1_mean + pc1_std, color=c, alpha=0.15)
ax.set_xlabel('Step t', fontsize=10)
ax.set_ylabel('PC1 value (mean ± std)', fontsize=10)
ax.set_title('h² PC1 trajectory by preference\n(Do different preferences diverge over time?)', fontsize=9)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
ax.set_xlim(0, MAX_STEPS - 1)

# --- (1,0) h² PCA 全データ（選好色分け）---
ax = fig.add_subplot(gs[1, 0])
for pref_idx, pref_name in enumerate(LANDMARK_NAMES):
    mask = (np.repeat(np.arange(4), N_EP_EACH * MAX_STEPS) == pref_idx)
    ax.scatter(X_all_2d[mask, 0], X_all_2d[mask, 1],
               c=PREF_COLORS[pref_name], s=1, alpha=0.15, label=pref_name)
    centroid = X_all_2d[mask].mean(axis=0)
    ax.scatter(*centroid, c=PREF_COLORS[pref_name], s=120, marker='*',
               edgecolors='k', linewidths=0.8, zorder=10)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title(f'h² PCA (all steps, acc={acc_all:.3f})', fontsize=9)
ax.legend(fontsize=7, markerscale=4)
ax.grid(True, alpha=0.2)

# --- (1,1) h² アトラクター軌跡 (代表エピソード) ---
ax = fig.add_subplot(gs[1, 1])
N_SHOW = min(8, N_EP_EACH)
for pref_idx, pref_name in enumerate(LANDMARK_NAMES):
    c = PREF_COLORS[pref_name]
    for ep in range(N_SHOW):
        traj = h2_traj_2d[pref_idx, ep]   # (MAX_STEPS, 2)
        ax.plot(traj[:, 0], traj[:, 1], color=c, lw=0.6, alpha=0.5)
        ax.scatter(traj[0, 0], traj[0, 1], c=c, s=15, marker='o', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c=c, s=20, marker='X', zorder=5)
    # ラベル用ダミー
    ax.plot([], [], color=c, lw=1.5, label=pref_name)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title(f'h² attractor trajectories ({N_SHOW} ep/pref)\n(o=start, X=end)', fontsize=9)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)

# --- (1,2) 混同行列 ---
ax = fig.add_subplot(gs[1, 2])
im = ax.imshow(cm_mat, cmap='Blues', aspect='auto')
ax.set_xticks(range(4)); ax.set_xticklabels(LANDMARK_NAMES, fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels(LANDMARK_NAMES, fontsize=8)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion matrix (acc={acc_all:.3f})', fontsize=9)
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(cm_mat[i, j]), ha='center', va='center',
                fontsize=9, color='white' if cm_mat[i, j] > cm_mat.max()*0.5 else 'black')
plt.colorbar(im, ax=ax, shrink=0.8)

# --- (1,3) h² PCA 最終ステップのみ ---
ax = fig.add_subplot(gs[1, 3])
for pref_idx, pref_name in enumerate(LANDMARK_NAMES):
    pts = h2_traj_2d[pref_idx, :, -1, :]   # (N_EP, 2) - 最終ステップ
    ax.scatter(pts[:, 0], pts[:, 1],
               c=PREF_COLORS[pref_name], s=20, alpha=0.6, label=pref_name)
    centroid = pts.mean(axis=0)
    ax.scatter(*centroid, c=PREF_COLORS[pref_name], s=150, marker='*',
               edgecolors='k', linewidths=0.8, zorder=10)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title(f'h² PCA at final step (t={MAX_STEPS-1})', fontsize=9)
ax.legend(fontsize=7, markerscale=2)
ax.grid(True, alpha=0.2)

# --- (2,0-1) h² PCA 時刻 0, 20, 40, 79 の分布変化 ---
timepoints = [0, int(MAX_STEPS*0.25), int(MAX_STEPS*0.5), MAX_STEPS-1]
for plot_i, t in enumerate(timepoints):
    ax = fig.add_subplot(gs[2, plot_i])
    for pref_idx, pref_name in enumerate(LANDMARK_NAMES):
        pts = h2_traj_2d[pref_idx, :, t, :]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=PREF_COLORS[pref_name], s=15, alpha=0.6, label=pref_name)
        ax.scatter(*pts.mean(axis=0), c=PREF_COLORS[pref_name],
                   s=100, marker='*', edgecolors='k', linewidths=0.7, zorder=10)
    ax.set_xlabel('PC1', fontsize=8); ax.set_ylabel('PC2', fontsize=8)
    step_acc = acc_h2[t]
    ax.set_title(f't={t}  acc={step_acc:.3f}', fontsize=9)
    if plot_i == 0:
        ax.legend(fontsize=6, markerscale=2)
    ax.grid(True, alpha=0.2)

plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

print("\n=== Phase 3 サマリ ===")
print(f"  MG r_mean          : {(r_x+r_y)/2:.4f}")
print(f"  h² acc (全)        : {acc_all:.4f}  (chance=0.25)")
print(f"  h²  acc peak       : {acc_h2.max():.4f}  (step {acc_h2.argmax()})")
print(f"  pos acc peak       : {acc_pos.max():.4f}  (step {acc_pos.argmax()})")
print(f"  om_pred acc peak   : {acc_om_pred.max():.4f}  (step {acc_om_pred.argmax()})")
print(f"\n  h² vs pos advantage (step 5):  {acc_h2[5]-acc_pos[5]:+.4f}")
print(f"  h² vs pos advantage (step 20): {acc_h2[20]-acc_pos[20]:+.4f}")
print(f"  h² vs pos advantage (step 40): {acc_h2[40]-acc_pos[40]:+.4f}")
print(f"\n  解釈:")
early_adv = acc_h2[5] - acc_pos[5]
if early_adv > 0.05:
    print(f"    h² は早期ステップで位置 baseline を上回る → 軌跡・方向の符号化あり")
elif early_adv > -0.03:
    print(f"    h² ≈ pos (早期) → 現在位置の追跡が主要な情報源")
else:
    print(f"    h² < pos (早期) → SM は位置を損失なく追跡できていない")
