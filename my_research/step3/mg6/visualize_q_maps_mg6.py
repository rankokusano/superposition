"""
visualize_q_maps_mg6.py

A-1 と A-2 の Q 値マップを 2 手法で作成して比較。

【手法1: ポリシー / MG 行動のみ】
  A-1: Actor の行動 → CriticLSTM(v, action) → Q → 次位置にプロット
  A-2: MG の推定行動 → VE(ov_enc, om_gen) → Q → 次位置にプロット

【手法2: 8 方向サンプリング】
  A-1: 8 方向均等サンプル → CriticLSTM(v, action_k) → Q → 各次位置にプロット
  A-2: 8 方向均等サンプル → VE(ov_enc, motion_k) → Q → 各次位置にプロット

【期待する結果】
  A-1 マップ: Red(-9,+9) 付近が高い、Green(-9,-9) 付近が低い  ← 学習済み Critic の確認
  A-2 マップ: Green(-9,-9) 付近が高い                         ← VE が A-2 の選好を学習していれば

実行コマンド（コンテナ内）:
    cd /work
    Xvfb :99 -screen 0 1024x768x24 & sleep 1
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/mg6/visualize_q_maps_mg6.py \\
        --run mg6_YYYYMMDD_HHMMSS \\
    2>&1 | tee my_research/step3/results/mg6_YYYYMMDD_HHMMSS/q_maps_log.txt

出力: {run_dir}/q_maps_mg6.png
"""

import sys, os, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from agent2_green import GreenFollowerAgent
from model_mg3 import SuperpositionNetworkWithMG3
from model import util as model_util

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()

RUN_DIR    = (args.run if os.path.isabs(args.run)
              else os.path.join(STEP3_DIR, 'results', args.run))
MODEL_FILE = os.path.join(RUN_DIR, 'model_mg6.pth')
SAVE_PATH  = os.path.join(RUN_DIR, 'q_maps_mg6.png')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

N_EPISODES = 50
MAX_STEPS  = 80
ARENA      = 9.0
GRID_N     = 20
CELL       = 2 * ARENA / GRID_N

# 8 方向の単位ベクトル（手法2 用）
N_DIRS = 8
SAMPLE_ACTIONS = np.array([
    (np.cos(2 * np.pi * k / N_DIRS), np.sin(2 * np.pi * k / N_DIRS))
    for k in range(N_DIRS)
], dtype=np.float32)

LANDMARKS = [('Red',   -9,  9,  'red'),
             ('Green', -9, -9,  'limegreen'),
             ('Blue',   9, -9,  'blue'),
             ('Cyan',   9,  9,  'cyan')]

# =========================================================================
# モデル読み込み
# =========================================================================
print(f"run: {RUN_DIR}")
ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
Q_SCALE   = ckpt.get('q_scale', 50.0)
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
green_follower = GreenFollowerAgent()

# =========================================================================
# NetDebug: forward pass を実行しつつ中間値 (ov_enc, om_gen, q_other) を保存
# =========================================================================
class NetDebug:
    """net.forward() を実行しつつ VE 入出力を外から参照できるようにするラッパー。"""
    def __init__(self, n):
        self.n = n
        self.last_ov_enc       = None
        self.last_om_generated = None
        self.last_q_other      = None

    def __call__(self, x, q_self, p1, p2):
        sv = x['self_vision']; sm = x['self_motion']
        sv_enc = self.n.self_vision_encoder_module(sv)
        ov_enc = self.n.other_vision_encoder_module(sv)
        om_gen = self.n.motion_generator(ov_enc)
        q_oth  = self.n.value_estimator(ov_enc, om_gen)
        self.last_ov_enc       = ov_enc.detach()
        self.last_om_generated = om_gen.detach()
        self.last_q_other      = q_oth.detach()
        sv_enc = model_util.mask(sv_enc, p1)
        ov_enc = model_util.mask(ov_enc, p2)
        ss, os = self.n.superposition_module(sv_enc, sm, q_self, ov_enc, om_gen, q_oth)
        so = self.n.integration_module(
            F.dropout(ss, p=0.5, training=False),
            F.dropout(os, p=0.5, training=False))
        pred = {'self_vision': self.n.vision_decoder_module(so)}
        return pred, ss, os

    def init_state(self, b): self.n.init_state(b)
    def detach_state(self): self.n.detach_state()

nd = NetDebug(net)

# =========================================================================
# 集計用アキュムレータ
# =========================================================================
def make_acc():
    return np.zeros((GRID_N, GRID_N)), np.zeros((GRID_N, GRID_N))

sum_a1_m1, cnt_a1_m1 = make_acc()
sum_a1_m2, cnt_a1_m2 = make_acc()
sum_a2_m1, cnt_a2_m1 = make_acc()
sum_a2_m2, cnt_a2_m2 = make_acc()

def record(q_sum, q_cnt, pos, q_val):
    j = int(np.clip((pos[0] + ARENA) / CELL, 0, GRID_N - 1))
    i = int(np.clip((pos[1] + ARENA) / CELL, 0, GRID_N - 1))
    q_sum[i, j] += q_val
    q_cnt[i, j] += 1

# =========================================================================
# データ収集
# =========================================================================
print(f"\nデータ収集中 ({N_EPISODES} ep x {MAX_STEPS} steps)...")

for ep in range(N_EPISODES):
    env.reset()
    nd.init_state(1)
    actor_h = critic_h = None
    v_raw, _, _, pos_a1, pos_a2 = env.step()

    for step in range(MAX_STEPS):
        v_t     = torch.FloatTensor(
            scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)
        v_raw_t = torch.FloatTensor(
            v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # A-1 の行動（Actor）
            action, _, actor_h = actor.sample(v_raw_t, actor_h)

            # Critic の LSTM 特徴量を一度だけ計算して Method1/2 で共用
            feat     = critic.encoder(v_raw_t).unsqueeze(1)
            lstm_out, critic_h = critic.lstm(feat, critic_h)
            lstm_out = lstm_out.squeeze(1)

            # ---- A-1 Method 1: ポリシー行動 --------------------------------
            q_policy = critic.q(torch.cat([lstm_out, action], dim=-1))
            q_a1_m1  = q_policy.item()
            next_a1_m1 = np.clip(pos_a1 + action.cpu().numpy()[0], -9.5, 9.5)
            record(sum_a1_m1, cnt_a1_m1, next_a1_m1, q_a1_m1)

            # ---- A-1 Method 2: 8 方向サンプリング --------------------------
            for act_vec in SAMPLE_ACTIONS:
                act_t = torch.FloatTensor([act_vec]).to(DEVICE)
                qk = critic.q(torch.cat([lstm_out, act_t], dim=-1)).item()
                record(sum_a1_m2, cnt_a1_m2,
                       np.clip(pos_a1 + act_vec, -9.5, 9.5), qk)

            # SM フォワードパス（SM + MG の状態更新 & 中間値を記録）
            q_self = q_policy / Q_SCALE
            p_mask = 0.0 if step == 0 else 1.0
            nd({'self_vision': v_t, 'self_motion': action}, q_self, 0.0, p_mask)
            nd.detach_state()

            # ---- A-2 Method 1: MG 推定行動 ----------------------------------
            om_np = nd.last_om_generated.cpu().numpy()[0]
            q_a2_m1 = nd.last_q_other.item()
            record(sum_a2_m1, cnt_a2_m1,
                   np.clip(pos_a2 + om_np, -9.5, 9.5), q_a2_m1)

            # ---- A-2 Method 2: 8 方向サンプリング ---------------------------
            ov_enc = nd.last_ov_enc
            for mot_vec in SAMPLE_ACTIONS:
                mot_t = torch.FloatTensor([mot_vec]).to(DEVICE)
                qk = net.value_estimator(ov_enc, mot_t).item()
                record(sum_a2_m2, cnt_a2_m2,
                       np.clip(pos_a2 + mot_vec, -9.5, 9.5), qk)

        # 環境ステップ
        a_np    = action.squeeze(0).cpu().numpy()
        om_true = green_follower.get_action(pos_a2)
        pos_a1  = np.clip(pos_a1 + a_np,    -9.5, 9.5)
        pos_a2  = np.clip(pos_a2 + om_true, -9.5, 9.5)
        env.self_agent.p  = pos_a1
        env.other_agent.p = pos_a2
        v_raw, _, _, _, _ = env.step()

    if (ep + 1) % 10 == 0:
        print(f"  {ep+1}/{N_EPISODES} done")

# =========================================================================
# ヒートマップ構築
# =========================================================================
def build_heatmap(q_sum, q_cnt):
    h = np.full((GRID_N, GRID_N), np.nan)
    mask = q_cnt > 0
    h[mask] = q_sum[mask] / q_cnt[mask]
    return h

h_a1_m1 = build_heatmap(sum_a1_m1, cnt_a1_m1)
h_a1_m2 = build_heatmap(sum_a1_m2, cnt_a1_m2)
h_a2_m1 = build_heatmap(sum_a2_m1, cnt_a2_m1)
h_a2_m2 = build_heatmap(sum_a2_m2, cnt_a2_m2)

print(f"\nカバレッジ (埋まったセル / {GRID_N*GRID_N} セル):")
print(f"  A-1 M1: {int((cnt_a1_m1>0).sum())}  A-1 M2: {int((cnt_a1_m2>0).sum())}")
print(f"  A-2 M1: {int((cnt_a2_m1>0).sum())}  A-2 M2: {int((cnt_a2_m2>0).sum())}")

for label, h in [('A-1 M1', h_a1_m1), ('A-1 M2', h_a1_m2),
                 ('A-2 M1', h_a2_m1), ('A-2 M2', h_a2_m2)]:
    valid = h[~np.isnan(h)]
    if len(valid) > 0:
        print(f"  {label}: min={valid.min():.3f}  max={valid.max():.3f}  mean={valid.mean():.3f}")

# =========================================================================
# 描画（2×2）
# =========================================================================
extent = [-ARENA, ARENA, -ARENA, ARENA]

def add_landmarks(ax):
    for name, lx, ly, lc in LANDMARKS:
        ax.plot(lx, ly, '*', color=lc, markersize=14,
                markeredgecolor='k', markeredgewidth=0.6, label=name)

# A-1 / A-2 ごとに色スケールを統一
vmin_a1 = np.nanmin([h_a1_m1, h_a1_m2])
vmax_a1 = np.nanmax([h_a1_m1, h_a1_m2])
vmin_a2 = np.nanmin([h_a2_m1, h_a2_m2])
vmax_a2 = np.nanmax([h_a2_m1, h_a2_m2])

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.suptitle(
    f'Q-value Maps  [mg6: {os.path.basename(RUN_DIR)}]\n'
    f'A-1 期待: Red(-9,+9) 付近が高い / A-2 期待: Green(-9,-9) 付近が高い',
    fontsize=12)

configs = [
    (axes[0, 0], 'A-1 Q-map  手法1: Actorポリシー行動', h_a1_m1, vmin_a1, vmax_a1),
    (axes[0, 1], 'A-2 Q-map  手法1: MG推定行動',        h_a2_m1, vmin_a2, vmax_a2),
    (axes[1, 0], 'A-1 Q-map  手法2: 8方向サンプリング', h_a1_m2, vmin_a1, vmax_a1),
    (axes[1, 1], 'A-2 Q-map  手法2: 8方向サンプリング', h_a2_m2, vmin_a2, vmax_a2),
]

for ax, title, h, vmin, vmax in configs:
    masked = np.ma.masked_invalid(h)
    im = ax.imshow(masked, origin='lower', extent=extent,
                   cmap='viridis', aspect='equal', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    add_landmarks(ax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X');  ax.set_ylabel('Y')

# 凡例は右上に一つだけ
axes[0, 1].legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")
