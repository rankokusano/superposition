"""
visualize_q_step2.py

Step2 学習結果の可視化
─────────────────────────────────────────────────
ValueEstimator が推測した Q(s,a) マップと
A-1 Critic の Q値マップを並べて比較する。

「ValueEstimator が緑ランドマーク付近を高く評価しているか」を確認する図。

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_q_step2.py

出力:
    step2/q_map_step2_value_estimator.png
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
import matplotlib.gridspec as gridspec

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
SAVE_PATH          = os.path.join(_HERE, 'q_map_step2_value_estimator.png')

GRID_SIZE    = 20   # グリッド解像度
WARMUP_STEPS = 3   # h² を育てるウォームアップステップ数
Q_SCALE      = 1.0  # 学習済みチェックポイントから上書きされる

# ランドマーク位置
LANDMARKS = {
    'Red (-9,9)':    (-9,  9, 'red'),
    'Green (-9,-9)': (-9, -9, 'limegreen'),
    'Blue (9,-9)':   ( 9, -9, 'blue'),
    'Cyan (9,9)':    ( 9,  9, 'cyan'),
}

# =========================================================================
# モデル読み込み
# =========================================================================
print(f"デバイス: {DEVICE}")

# --- Step2 モデル ---
model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithQ(model_config).to(DEVICE)

if not os.path.exists(MODEL_SAVE_PATH):
    print(f"[WARNING] {MODEL_SAVE_PATH} が見つかりません。")
    print("  学習前の初期状態で可視化します (ランダム初期化)。")
    ckpt_iizuka = torch.load(
        os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth'),
        map_location=DEVICE
    )
    net.load_iizuka_weights(ckpt_iizuka, DEVICE)
else:
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    net.load_state_dict(ckpt['net_state_dict'])
    Q_SCALE    = ckpt.get('q_scale', 1.0)   # 学習時と同じスケールで正規化
    final_loss = ckpt.get('final_loss', float('nan'))
    print(f"学習済みモデルをロード: episodes={ckpt.get('episodes','?')}, "
          f"final_loss={final_loss:.6f}, Q_SCALE={Q_SCALE}")

net.eval()

# --- A-1 Critic (比較用) ---
critic = CriticLSTM().to(DEVICE)
critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
critic.eval()

# --- Actor (ウォームアップ用) ---
actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()

# --- 環境 ---
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

# 3つのマップを計算
q_map_ve_init   = np.zeros((GRID_SIZE, GRID_SIZE))  # ValueEstimator (h²=zeros)
q_map_ve_warmup = np.zeros((GRID_SIZE, GRID_SIZE))  # ValueEstimator (warmup後)
q_map_critic    = np.zeros((GRID_SIZE, GRID_SIZE))  # A-1 Critic (比較用)

A1_REF_POS = np.array([0.0, 0.0])  # A-1 の固定参照位置

print(f"\nQ値マップ計算中 ({GRID_SIZE}×{GRID_SIZE} グリッド)...")

for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        a2_pos = np.array([x, y])
        om_np  = green_follower.get_action(a2_pos)
        om_t   = torch.FloatTensor(om_np).unsqueeze(0).to(DEVICE)

        # --- A-1 の視覚 (A-1 は参照位置に固定) ---
        v_raw = env.capture_at_pos(A1_REF_POS, a2_pos)
        v_t   = torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # ── (1) ValueEstimator: h²=zeros の初期状態 ──────────────────
            net.superposition_module.init_state(1)
            h2_init = net.superposition_module.state['other'].hidden  # zeros
            q_ve_init = net.value_estimator(h2_init, om_t)
            q_map_ve_init[j, i] = q_ve_init.item()

            # ── (2) ValueEstimator: ウォームアップ後 ─────────────────────
            net.superposition_module.init_state(1)
            actor_h = None
            v_tmp = v_t
            for _ in range(WARMUP_STEPS):
                a_tmp, _, actor_h = actor.sample(v_tmp, actor_h)
                a_np_tmp = a_tmp.squeeze(0).cpu().numpy()
                sm_tmp = torch.FloatTensor(a_np_tmp).unsqueeze(0).to(DEVICE)
                x_in = {
                    'self_vision':  v_tmp,
                    'self_motion':  sm_tmp,
                    'other_motion': om_t,
                }
                # q_self は Critic から取得 (学習時と同じ Q_SCALE で正規化)
                q_s, _ = critic(v_tmp, a_tmp)
                q_s_scaled = q_s / Q_SCALE
                _, _, _ = net(x_in, q_s_scaled, 0.0, 0.0)  # hidden state を更新

            h2_warmed = net.superposition_module.state['other'].hidden
            q_ve_warm = net.value_estimator(h2_warmed, om_t)
            q_map_ve_warmup[j, i] = q_ve_warm.item()

            # ── (3) A-1 Critic (比較用) ──────────────────────────────────
            action_ref, _, _ = actor.sample(v_t)
            q_c, _ = critic(v_t, action_ref)
            q_map_critic[j, i] = q_c.item()

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{GRID_SIZE} 列完了")

print("計算完了")

# =========================================================================
# 可視化
# =========================================================================
fig = plt.figure(figsize=(18, 6))
fig.suptitle('Step2: Q-value Map Comparison\n'
             '(A-2 = GreenFollower → target: green landmark at (-9,-9))',
             fontsize=13)

gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

titles = [
    f'ValueEstimator Q̂(s,a)\n(h²=zeros, initial)',
    f'ValueEstimator Q̂(s,a)\n(after {WARMUP_STEPS}-step warmup)',
    'A-1 Critic Q(s,a)\n(reference)',
]
maps = [q_map_ve_init, q_map_ve_warmup, q_map_critic]

for col, (title, qmap) in enumerate(zip(titles, maps)):
    ax = fig.add_subplot(gs[col])
    vabs = np.abs(qmap).max()
    vmax = max(vabs, 1e-6)

    im = ax.imshow(
        qmap,
        extent=[-9, 9, -9, 9],
        origin='lower',
        cmap='RdYlGn',
        vmin=-vmax,
        vmax=vmax,
        aspect='equal',
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label='Q value')

    # ランドマーク
    for label, (lx, ly, lc) in LANDMARKS.items():
        ax.plot(lx, ly, '*', color=lc, markersize=14,
                markeredgecolor='black', markeredgewidth=0.5,
                label=label.split(' ')[0])

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.legend(fontsize=7, loc='upper right')

plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存完了: {SAVE_PATH}")

# ── 数値サマリー ──────────────────────────────────────────────
GREEN_IDX_X = int(np.argmin(np.abs(xs - (-9))))
GREEN_IDX_Y = int(np.argmin(np.abs(ys - (-9))))
RED_IDX_X   = int(np.argmin(np.abs(xs - (-9))))
RED_IDX_Y   = int(np.argmin(np.abs(ys -  (9))))

print("\n=== 重要座標でのQ値 ===")
for map_name, qmap in [('VE (init)', q_map_ve_init),
                        ('VE (warmup)', q_map_ve_warmup),
                        ('Critic', q_map_critic)]:
    q_green = qmap[GREEN_IDX_Y, GREEN_IDX_X]
    q_red   = qmap[RED_IDX_Y,   RED_IDX_X]
    q_center = qmap[GRID_SIZE//2, GRID_SIZE//2]
    print(f"  [{map_name}] "
          f"Green(-9,-9)={q_green:.4f}  "
          f"Red(-9,9)={q_red:.4f}  "
          f"Center(0,0)={q_center:.4f}")
