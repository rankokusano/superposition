"""
visualize_pref_pca_mg7.py

mg7 モデルの Shared Module 隠れ状態 h² を、
A-2 の選好（Red/Green/Blue/Cyan）でカラーリングして PCA 可視化。

【元論文との対応】
    Noguchi et al. (2022) Fig.6 では Motion Generator の内部状態を
    行動パターン（CW/CCW/停止）でカラーリングし、選好に応じたアトラクターが
    形成されることを示した。
    本スクリプトは同様の手法で「4 種の選好に対して h² が分離するか」を確認する。

【期待する結果】
    各選好のデータ点が PCA 空間でクラスタを形成していれば、
    SM が選好情報を h² に符号化していると言える。

実行コマンド（コンテナ内）:
    cd /work
    Xvfb :99 -screen 0 1024x768x24 & sleep 1
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/mg7/visualize_pref_pca_mg7.py \\
        --run mg7_YYYYMMDD_HHMMSS \\
    2>&1 | tee my_research/step3/results/mg7_YYYYMMDD_HHMMSS/pref_pca_log.txt

出力: {run_dir}/pref_pca_mg7.png
"""

import sys, os, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

_HERE       = os.path.dirname(os.path.abspath(__file__))
STEP3_DIR   = os.path.abspath(os.path.join(_HERE, '..'))
STEP2_DIR   = os.path.abspath(os.path.join(_HERE, '../../step2'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP3_DIR, STEP2_DIR, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from util import scale_vision
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_multi_pref import LandmarkFollowerAgent, LANDMARK_NAMES
from model_mg3 import SuperpositionNetworkWithMG3

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()

RUN_DIR    = (args.run if os.path.isabs(args.run)
              else os.path.join(STEP3_DIR, 'results', args.run))
MODEL_FILE = os.path.join(RUN_DIR, 'model_mg7.pth')
SAVE_PATH  = os.path.join(RUN_DIR, 'pref_pca_mg7.png')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

N_EPISODES_PER_PREF = 25
MAX_STEPS = 80

# 選好ごとの表示色（ランドマーク色に合わせる）
PREF_COLORS = {
    'Red':   'red',
    'Green': 'limegreen',
    'Blue':  'blue',
    'Cyan':  'cyan',
}

# =========================================================================
# モデル読み込み
# =========================================================================
print(f"run: {RUN_DIR}")
ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
Q_SCALE   = ckpt.get('q_scale',   50.0)
mg_hidden = ckpt.get('mg_hidden', 64)
print(f"  episodes={ckpt.get('episodes','?')}  "
      f"final_loss={ckpt.get('final_loss', float('nan')):.5f}")

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

# =========================================================================
# 選好ごとに h² を収集
# =========================================================================
all_h2     = []   # 全 h² (N_total x hidden_dim)
all_labels = []   # 対応する選好ラベル文字列

print(f"\n選好ごとに h² を収集中 ({N_EPISODES_PER_PREF} ep × 4 選好)...")

for pref_name in LANDMARK_NAMES:
    follower = LandmarkFollowerAgent(pref_name)
    ep_count = 0

    for ep in range(N_EPISODES_PER_PREF):
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
                q_raw,  critic_h   = critic(v_raw_t, action, critic_h)
                q_self             = q_raw / Q_SCALE

                _, h1, h2 = net(
                    {'self_vision': v_t, 'self_motion': action},
                    q_self, 0.0, 0.0
                )
                net.detach_state()

            all_h2.append(h2.squeeze(0).cpu().numpy())
            all_labels.append(pref_name)

            a_np      = action.squeeze(0).cpu().numpy()
            om_np     = follower.get_action(other_pos)
            self_pos  = np.clip(self_pos  + a_np,  -9.5, 9.5)
            other_pos = np.clip(other_pos + om_np, -9.5, 9.5)
            env.self_agent.p  = self_pos
            env.other_agent.p = other_pos
            v_raw, _, _, _, _ = env.step()

        ep_count += 1

    print(f"  {pref_name}: {ep_count * MAX_STEPS} steps 収集完了")

H2     = np.array(all_h2)
labels = np.array(all_labels)
print(f"\n合計: {len(H2)} ステップ  (hidden_dim={H2.shape[1]})")

# =========================================================================
# PCA（h² 全体で fitting → 同一空間に 4 選好を投影）
# =========================================================================
pca    = PCA(n_components=2)
H2_2d  = pca.fit_transform(H2)
var    = pca.explained_variance_ratio_
print(f"PCA 寄与率: PC1={var[0]:.3f}  PC2={var[1]:.3f}  合計={var.sum():.3f}")

# =========================================================================
# クラスタ分離の定量評価
# =========================================================================
print("\n=== クラスタ分離の定量評価 ===")
centers = {}
for pref in LANDMARK_NAMES:
    mask = labels == pref
    centers[pref] = H2_2d[mask].mean(axis=0)
    std = H2_2d[mask].std(axis=0).mean()
    print(f"  {pref:6s}: center=({centers[pref][0]:+.3f}, {centers[pref][1]:+.3f})  "
          f"within-std={std:.3f}")

print("\n  クラスタ間距離（重心間のユークリッド距離）:")
for i, p1 in enumerate(LANDMARK_NAMES):
    for p2 in LANDMARK_NAMES[i+1:]:
        d = np.linalg.norm(centers[p1] - centers[p2])
        print(f"    {p1} - {p2}: {d:.3f}")

# 分離スコア: 全クラスタ間の平均距離 / 全体の within-std 平均
all_within_std = np.mean([H2_2d[labels == p].std(axis=0).mean() for p in LANDMARK_NAMES])
dists = [np.linalg.norm(centers[p1] - centers[p2])
         for i, p1 in enumerate(LANDMARK_NAMES)
         for p2 in LANDMARK_NAMES[i+1:]]
sep_score = np.mean(dists) / (all_within_std + 1e-8)
print(f"\n  分離スコア (クラスタ間距離平均 / within-std): {sep_score:.3f}")
print(f"  （目安: >1.0 なら分離傾向あり、>2.0 なら明確なクラスタ）")

# =========================================================================
# 描画
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f'h² colored by A-2 preference  [mg7: {os.path.basename(RUN_DIR)}]\n'
    f'PC1={var[0]:.2f}, PC2={var[1]:.2f}  '
    f'({N_EPISODES_PER_PREF} ep × 4 prefs × {MAX_STEPS} steps)',
    fontsize=11
)

# --- 左: 選好ラベルで色付け（メイン） ---
ax = axes[0]
for pref in LANDMARK_NAMES:
    mask = labels == pref
    ax.scatter(H2_2d[mask, 0], H2_2d[mask, 1],
               c=PREF_COLORS[pref], s=4, alpha=0.4, label=pref)
    # 重心をマーク
    cx, cy = centers[pref]
    ax.plot(cx, cy, '*', color=PREF_COLORS[pref], markersize=16,
            markeredgecolor='black', markeredgewidth=1.0)

ax.set_title('Process-2  h²  (colored by A-2 preference)\n'
             '★ = cluster center', fontsize=10)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.2)

# --- 右: 重心だけを拡大表示（分離の様子を確認しやすく） ---
ax2 = axes[1]
for pref in LANDMARK_NAMES:
    cx, cy = centers[pref]
    ax2.plot(cx, cy, 'o', color=PREF_COLORS[pref], markersize=18,
             markeredgecolor='black', markeredgewidth=1.2)
    ax2.annotate(pref, (cx, cy), textcoords='offset points',
                 xytext=(8, 4), fontsize=12, color=PREF_COLORS[pref],
                 fontweight='bold')

# 重心間の線を描画
for i, p1 in enumerate(LANDMARK_NAMES):
    for p2 in LANDMARK_NAMES[i+1:]:
        x1, y1 = centers[p1]
        x2, y2 = centers[p2]
        ax2.plot([x1, x2], [y1, y2], 'k--', alpha=0.3, linewidth=0.8)

ax2.set_title(f'Cluster centers only\nsep_score={sep_score:.2f}  '
              f'(>2.0 = clear separation)', fontsize=10)
ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")
