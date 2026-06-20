"""
visualize_q_final.py

Step2 学習結果の可視化（2パネル・整理版）
─────────────────────────────────────────────────
【左パネル】 ValueEstimator が推測するA-2のQ値マップ
    A-2の位置を20×20グリッドで変化させる。
    各位置で Φ_s を1ステップ通してh²を更新し、
    ValueEstimator(h², A-2の緑向き運動) を出力する。
    → これが「Process-2が学習した価値観マップ」

【右パネル】 A-1 Critic の実際のQ値マップ（比較用）
    A-1の位置を20×20グリッドで変化させる。
    各位置で Critic(視覚, A-1の行動) を計算する。
    → A-1が赤付近で高いはず（Step1で確認済み）

研究の問い：
    左パネルが「緑付近で高い」 → A-2の価値観を獲得できた
    右パネルが「赤付近で高い」 → A-1の価値観（既知・確認用）

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_q_final.py

出力:
    step2/q_map_final.png
─────────────────────────────────────────────────
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
# 設定
# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH    = os.path.join(_HERE, 'model_q.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')
SAVE_PATH          = os.path.join(_HERE, 'q_map_final.png')

GRID_SIZE = 20

LANDMARKS = [
    ('Red',   -9,  9, 'red'),
    ('Green', -9, -9, 'limegreen'),
    ('Blue',   9, -9, 'blue'),
    ('Cyan',   9,  9, 'cyan'),
]

Q_SCALE = 1.0  # チェックポイントから上書きされる

# =========================================================================
# モデル読み込み
# =========================================================================
print(f"デバイス: {DEVICE}")

model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithQ(model_config).to(DEVICE)

if not os.path.exists(MODEL_SAVE_PATH):
    print(f"[WARNING] {MODEL_SAVE_PATH} が見つかりません。初期状態で可視化します。")
    iizuka_ckpt = torch.load(
        os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth'),
        map_location=DEVICE
    )
    net.load_iizuka_weights(iizuka_ckpt, DEVICE)
else:
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt['net_state_dict'])
    Q_SCALE = ckpt.get('q_scale', 1.0)
    print(f"学習済みモデルをロード: "
          f"episodes={ckpt.get('episodes','?')}, "
          f"final_loss={ckpt.get('final_loss', float('nan')):.5f}, "
          f"Q_SCALE={Q_SCALE}")

net.eval()

critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
critic.eval()

actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()

env_config = load_config(ENV_CONFIG_PATH)
env = creator.create_environment(env_config.environment)
env.init()
env.off_display()

green_follower = GreenFollowerAgent()

# =========================================================================
# Q値マップ計算
# =========================================================================
xs = np.linspace(-9, 9, GRID_SIZE)
ys = np.linspace(-9, 9, GRID_SIZE)

q_map_ve     = np.zeros((GRID_SIZE, GRID_SIZE))  # 左: ValueEstimator
q_map_critic = np.zeros((GRID_SIZE, GRID_SIZE))  # 右: A-1 Critic

# 右パネル用: A-2は緑ランドマークに固定
A2_FIXED_POS = np.array([-9.0, -9.0])
# 左パネル用: A-1は中央に固定
A1_FIXED_POS = np.array([0.0,  0.0])

print(f"\nQ値マップ計算中 ({GRID_SIZE}×{GRID_SIZE} グリッド)...")

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        pos = np.array([x, y])
        with torch.no_grad():

            # ──────────────────────────────────────────────────────────────
            # 【左パネル】ValueEstimator Q̂: A-2の各位置でのQ推測
            #
            # 「学習済みネットワークが各位置でどんなQ値を推測するか」を示す。
            #
            # 手順:
            #   1. A-2をグリッド位置 (x,y) に置く。A-1は中央固定。
            #   2. その視覚シーンを Φ_s に1ステップ通す → h²が更新される
            #      (h²=zeros から始めて「この位置を見た後の状態」にする)
            #   3. ValueEstimator(h², 緑向き運動) を計算 → Q値
            #
            # h²=zeros のまま出力しない理由:
            #   「何も見ていない状態」では視覚情報がh²に入っていないため、
            #   位置による違いが運動ベクトルの差だけになってしまう。
            #   1ステップ通すことで「この位置の景色を見た後の推測」になる。
            # ──────────────────────────────────────────────────────────────
            v_raw = env.capture_at_pos(A1_FIXED_POS, pos)
            v_t   = torch.FloatTensor(
                v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            om_np = green_follower.get_action(pos)
            om_t  = torch.FloatTensor(om_np).unsqueeze(0).to(DEVICE)

            action_a1, _, _ = actor.sample(v_t)
            q_self_raw, _   = critic(v_t, action_a1)
            q_self = q_self_raw / Q_SCALE

            net.superposition_module.init_state(1)
            x_in = {
                'self_vision':  v_t,
                'self_motion':  action_a1,
                'other_motion': om_t,
            }
            _, _, h2 = net(x_in, q_self, 0.0, 0.0)

            q_ve = net.value_estimator(h2, om_t)
            q_map_ve[j, i] = q_ve.item()

            # ──────────────────────────────────────────────────────────────
            # 【右パネル】A-1 Critic Q: A-1の各位置でのQ値
            #
            # A-1をグリッド位置 (x,y) に置く。A-2は緑ランドマーク固定。
            # Criticが出力するQ値をそのまま記録。
            # ──────────────────────────────────────────────────────────────
            v_raw_a1 = env.capture_at_pos(pos, A2_FIXED_POS)
            v_t_a1   = torch.FloatTensor(
                v_raw_a1.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            action_a1_ref, _, _ = actor.sample(v_t_a1)
            q_c, _ = critic(v_t_a1, action_a1_ref)
            q_map_critic[j, i] = q_c.item()

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{GRID_SIZE} 列完了")

print("計算完了")

# =========================================================================
# 可視化
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(
    'Step2: Learned Q-value Maps\n'
    'Left: Process-2 Q̂ (ValueEstimator) — does A-2 prefer Green?\n'
    'Right: A-1 Critic Q — reference, should prefer Red',
    fontsize=11
)

panels = [
    (axes[0], q_map_ve,
     'ValueEstimator Q̂(s,a)  [Process-2]\n'
     'A-2の各位置でΦ_sを1ステップ通した後のQ推測\n'
     '← 緑付近が高ければ成功'),
    (axes[1], q_map_critic,
     'A-1 Critic Q(s,a)  [Reference]\n'
     'A-1の各位置でのCritic出力\n'
     '← 赤付近が高いはず'),
]

for ax, qmap, title in panels:
    vabs = max(np.abs(qmap).max(), 1e-6)
    im = ax.imshow(
        qmap,
        extent=[-9, 9, -9, 9],
        origin='lower',
        cmap='RdYlGn',
        vmin=-vabs,
        vmax=vabs,
        aspect='equal',
    )
    plt.colorbar(im, ax=ax, shrink=0.85, label='Q value')

    for name, lx, ly, lc in LANDMARKS:
        ax.plot(lx, ly, '*', color=lc, markersize=14,
                markeredgecolor='black', markeredgewidth=0.5, label=name)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存完了: {SAVE_PATH}")

# =========================================================================
# 数値サマリー
# =========================================================================
def idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))

print("\n=== 重要座標でのQ値 ===")
print(f"{'':22s}  Green(-9,-9)  Red(-9,+9)  Center(0,0)")
for name, qmap in [('ValueEstimator (左)', q_map_ve),
                   ('A-1 Critic    (右)', q_map_critic)]:
    q_g = qmap[idx(ys, -9), idx(xs, -9)]
    q_r = qmap[idx(ys,  9), idx(xs, -9)]
    q_c = qmap[idx(ys,  0), idx(xs,  0)]
    print(f"  {name}:  {q_g:+.4f}        {q_r:+.4f}      {q_c:+.4f}")
