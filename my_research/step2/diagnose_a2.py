"""
diagnose_a2.py

A-2の動作とValueEstimatorの出力を1エピソード追って診断する。

確認したいこと:
    1. A-2は本当に緑(-9,-9)に向かって動いているか
    2. ValueEstimatorのQ値はA-2が緑に近づくにつれて変化するか
    3. A-1の視野にA-2が見えているか（位置によって視覚が変わるか）

出力:
    step2/diagnose_trajectory.png  ... A-2の軌跡 + Q値の時系列
    step2/diagnose_vision_*.png    ... A-1の視覚サンプル（数ステップ分）

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 diagnose_a2.py
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '../../'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../'))
for p in [_HERE, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_green import GreenFollowerAgent
from model_q import SuperpositionNetworkWithQ

# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH    = os.path.join(_HERE, 'model_q.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

MAX_STEPS   = 60
A2_START    = np.array([8.0, 8.0])  # 緑から一番遠い角（右上）から開始

Q_SCALE = 1.0

# =========================================================================
# モデル読み込み
# =========================================================================
model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithQ(model_config).to(DEVICE)

if os.path.exists(MODEL_SAVE_PATH):
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt['net_state_dict'])
    Q_SCALE = ckpt.get('q_scale', 1.0)
    print(f"学習済みモデルをロード (episodes={ckpt.get('episodes','?')}, "
          f"loss={ckpt.get('final_loss', float('nan')):.5f})")
else:
    print("[WARNING] model_q.pth なし。初期状態で診断します。")
    iizuka_ckpt = torch.load(
        os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth'),
        map_location=DEVICE)
    net.load_iizuka_weights(iizuka_ckpt, DEVICE)

net.eval()

actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()

critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
critic.eval()

env_config = load_config(ENV_CONFIG_PATH)
env = creator.create_environment(env_config.environment)
env.init()
env.off_display()

green_follower = GreenFollowerAgent()
GREEN_POS = np.array([-9.0, -9.0])

# =========================================================================
# 1エピソード実行して記録
# =========================================================================
print(f"\n診断エピソード開始 (A-2スタート位置: {A2_START})")
print(f"Q_SCALE = {Q_SCALE}")

env.reset()
net.superposition_module.init_state(1)

# A-2を指定位置にセット
env.other_agent.p = A2_START.copy()
other_pos = A2_START.copy()

# A-1は中央付近からランダムスタート
env.self_agent.p = np.array([0.0, 0.0])
self_pos = np.array([0.0, 0.0])

v_raw, _, _, _, _ = env.step()
actor_hidden  = None
critic_hidden = None

# 記録用バッファ
record_a2_pos  = []
record_dist    = []
record_q_ve    = []
record_q_critic = []
vision_samples  = []   # 数ステップの視覚画像を保存
VISION_SAVE_STEPS = [0, 10, 20, 40]

for step in range(MAX_STEPS):
    v_t = torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    # A-1の行動
    with torch.no_grad():
        action, _, actor_hidden = actor.sample(v_t, actor_hidden)
    a_np = action.squeeze(0).cpu().numpy()

    # A-1のQ値（正規化）
    with torch.no_grad():
        q_raw, critic_hidden = critic(v_t, action, critic_hidden)
    q_self = q_raw / Q_SCALE

    # A-2の運動
    om_np = green_follower.get_action(other_pos)
    om_t  = torch.FloatTensor(om_np).unsqueeze(0).to(DEVICE)
    sm_t  = torch.FloatTensor(a_np).unsqueeze(0).to(DEVICE)

    # Φ_sを通す
    x_in = {'self_vision': v_t, 'self_motion': sm_t, 'other_motion': om_t}
    with torch.no_grad():
        _, _, h2 = net(x_in, q_self, 0.0, 0.0)
        q_ve = net.value_estimator(h2, om_t)

    # 記録
    dist = float(np.linalg.norm(other_pos - GREEN_POS))
    record_a2_pos.append(other_pos.copy())
    record_dist.append(dist)
    record_q_ve.append(float(q_ve.item()))
    record_q_critic.append(float(q_raw.item()))

    if step in VISION_SAVE_STEPS:
        vision_samples.append((step, v_raw.copy(), other_pos.copy()))

    # 位置更新
    new_self_pos  = np.clip(self_pos + a_np,  -9.5, 9.5)
    new_other_pos = np.clip(other_pos + om_np, -9.5, 9.5)
    env.self_agent.p  = new_self_pos
    env.other_agent.p = new_other_pos
    v_next_raw, _, _, _, _ = env.step()
    net.superposition_module.detach_state()

    v_raw     = v_next_raw
    self_pos  = new_self_pos
    other_pos = new_other_pos

print(f"A-2の最終位置: {other_pos}  (緑との距離: {np.linalg.norm(other_pos - GREEN_POS):.2f})")
print(f"ValueEstimator Q範囲: [{min(record_q_ve):.4f}, {max(record_q_ve):.4f}]")

# =========================================================================
# 可視化
# =========================================================================

fig = plt.figure(figsize=(16, 10))
fig.suptitle('A-2 Trajectory Diagnosis\n'
             f'A-2 start: {A2_START}, target: Green(-9,-9)', fontsize=12)

# ── (1) A-2の軌跡 ───────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
traj = np.array(record_a2_pos)
sc = ax1.scatter(traj[:, 0], traj[:, 1],
                 c=np.arange(len(traj)), cmap='plasma', s=20, zorder=3)
ax1.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.3, linewidth=0.8)
ax1.plot(traj[0, 0], traj[0, 1], 'ko', markersize=8, label='Start', zorder=5)
ax1.plot(traj[-1, 0], traj[-1, 1], 'k^', markersize=8, label='End', zorder=5)
plt.colorbar(sc, ax=ax1, label='step')
for name, lx, ly, lc in [('Red', -9, 9, 'red'), ('Green', -9, -9, 'limegreen'),
                           ('Blue', 9, -9, 'blue'), ('Cyan', 9, 9, 'cyan')]:
    ax1.plot(lx, ly, '*', color=lc, markersize=14,
             markeredgecolor='black', markeredgewidth=0.5, label=name)
ax1.set_xlim(-10, 10)
ax1.set_ylim(-10, 10)
ax1.set_title('A-2 の軌跡\n(緑に向かっているか確認)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# ── (2) 緑との距離（時系列） ─────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(record_dist, 'g-o', markersize=3, linewidth=1.5)
ax2.axhline(1.0, color='gray', linestyle='--', label='停止閾値 (1.0)')
ax2.set_title('緑(-9,-9)との距離（時系列）\n(単調減少していれば正常)')
ax2.set_xlabel('step')
ax2.set_ylabel('distance to green')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── (3) ValueEstimatorのQ値（時系列） ──────────────────────
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(record_q_ve, 'b-o', markersize=3, linewidth=1.5, label='ValueEstimator Q̂')
ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax3.set_title('ValueEstimator Q̂ の時系列\n(緑に近づくにつれ上がるか？)')
ax3.set_xlabel('step')
ax3.set_ylabel('estimated Q')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ── (4) Q値 vs 緑との距離（散布図） ─────────────────────────
ax4 = fig.add_subplot(2, 3, 4)
ax4.scatter(record_dist, record_q_ve, c=np.arange(len(record_dist)),
            cmap='plasma', s=20)
ax4.set_title('Q̂ vs 緑との距離\n(右下から左上への傾きがあれば成功)')
ax4.set_xlabel('distance to green')
ax4.set_ylabel('estimated Q̂')
ax4.grid(True, alpha=0.3)

# 相関係数を計算
corr = float(np.corrcoef(record_dist, record_q_ve)[0, 1])
ax4.set_title(f'Q̂ vs 緑との距離  (相関 r={corr:.3f})\n'
              f'(r<0 なら「近い=高Q」= 成功)')

# ── (5) A-1の視覚サンプル ────────────────────────────────────
for k, (step_idx, v_img, a2_p) in enumerate(vision_samples[:3]):
    ax = fig.add_subplot(2, 3, 5 + k) if k < 1 else None
    if k == 0:
        ax = fig.add_subplot(2, 3, 5)
    elif k == 1:
        ax = fig.add_subplot(2, 3, 6)
    else:
        break
    if ax is not None:
        ax.imshow(v_img)
        dist_v = float(np.linalg.norm(a2_p - GREEN_POS))
        ax.set_title(f'A-1の視覚 (step={step_idx})\n'
                     f'A-2位置={a2_p.round(1)}, 緑距離={dist_v:.1f}')
        ax.axis('off')

plt.tight_layout()
save_path = os.path.join(_HERE, 'diagnose_trajectory.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n保存完了: {save_path}")

# =========================================================================
# テキスト診断
# =========================================================================
print("\n=== 診断結果 ===")
dist_start = record_dist[0]
dist_end   = record_dist[-1]
print(f"  A-2開始距離: {dist_start:.2f}  →  終了距離: {dist_end:.2f}  "
      f"({'✓ 緑に近づいた' if dist_end < dist_start else '✗ 近づいていない'})")
print(f"  Q値と距離の相関: r = {corr:.4f}  "
      f"({'✓ 近いほど高い (r<0)' if corr < -0.1 else '✗ 相関なし'})")
q_first10 = np.mean(record_q_ve[:10])
q_last10  = np.mean(record_q_ve[-10:])
print(f"  序盤Q平均={q_first10:.4f}  終盤Q平均={q_last10:.4f}  "
      f"({'✓ 近づくにつれ上昇' if q_last10 > q_first10 else '✗ 変化なし/下降'})")
