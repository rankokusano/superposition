"""
visualize_q_maps_mg7.py

mg7 モデルの Q 値マップを選好別に可視化。

【mg6 との違い】
    A-2 の評価を選好（Red/Green/Blue/Cyan）ごとに分けて行い、
    「VE が A-2 の現在の選好に応じた Q 値を出力できているか」を検証する。

【レイアウト（3×2 グリッド）】
    (0,0) A-1 Q-map（8方向サンプリング）  ← 学習済みCriticの参照用
    (0,1) A-2 Q-map [Green 追従評価]
    (1,0) A-2 Q-map [Red 追従評価]
    (1,1) A-2 Q-map [Blue 追従評価]
    (2,0) A-2 Q-map [Cyan 追従評価]
    (2,1) (空白)

【期待する結果】
    各 A-2 マップで「対応するランドマーク付近の Q 値が高い」ならば、
    VE が選好推論を学習できていると言える。
    例: Green 追従評価 → Green(-9,-9) 付近が高い

実行コマンド（コンテナ内）:
    cd /work
    Xvfb :99 -screen 0 1024x768x24 & sleep 1
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/mg7/visualize_q_maps_mg7.py \\
        --run mg7_YYYYMMDD_HHMMSS \\
    2>&1 | tee my_research/step3/results/mg7_YYYYMMDD_HHMMSS/q_maps_log.txt

出力: {run_dir}/q_maps_mg7.png
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
from agent2_multi_pref import LandmarkFollowerAgent, LANDMARK_POSITIONS, LANDMARK_NAMES
from model_mg3 import SuperpositionNetworkWithMG3
from model import util as model_util

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()

RUN_DIR    = (args.run if os.path.isabs(args.run)
              else os.path.join(STEP3_DIR, 'results', args.run))
MODEL_FILE = os.path.join(RUN_DIR, 'model_mg7.pth')
SAVE_PATH  = os.path.join(RUN_DIR, 'q_maps_mg7.png')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

N_EPISODES_PER_PREF = 25   # 選好ごとのエピソード数（計 4×25 = 100）
MAX_STEPS  = 80
ARENA      = 9.0
GRID_N     = 20
CELL       = 2 * ARENA / GRID_N

N_DIRS = 8
SAMPLE_ACTIONS = np.array([
    (np.cos(2 * np.pi * k / N_DIRS), np.sin(2 * np.pi * k / N_DIRS))
    for k in range(N_DIRS)
], dtype=np.float32)

LANDMARK_INFO = [
    ('Red',   -9,  9, 'red'),
    ('Green', -9, -9, 'limegreen'),
    ('Blue',   9, -9, 'blue'),
    ('Cyan',   9,  9, 'cyan'),
]

# =========================================================================
# モデル読み込み
# =========================================================================
print(f"run: {RUN_DIR}")
ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
Q_SCALE   = ckpt.get('q_scale', 50.0)
mg_hidden = ckpt.get('mg_hidden', 64)
print(f"  episodes={ckpt.get('episodes','?')}  "
      f"final_loss={ckpt.get('final_loss', float('nan')):.5f}")
print(f"  pref_dist={ckpt.get('pref_dist', 'N/A')}")

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
# NetDebug: MG / VE 中間値を外から参照するラッパー
# =========================================================================
class NetDebug:
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

def record(q_sum, q_cnt, pos, q_val):
    j = int(np.clip((pos[0] + ARENA) / CELL, 0, GRID_N - 1))
    i = int(np.clip((pos[1] + ARENA) / CELL, 0, GRID_N - 1))
    q_sum[i, j] += q_val
    q_cnt[i, j] += 1

def build_heatmap(q_sum, q_cnt):
    h = np.full((GRID_N, GRID_N), np.nan)
    mask = q_cnt > 0
    h[mask] = q_sum[mask] / q_cnt[mask]
    return h


# =========================================================================
# A-1 Q-map（選好に依存しないため一度だけ収集）
# =========================================================================
print(f"\nA-1 Q-map 収集中 ({N_EPISODES_PER_PREF} ep)...")
sum_a1, cnt_a1 = make_acc()
follower_ref = LandmarkFollowerAgent('Green')   # A-2 の行動は A-1 Q-map に影響しない

for ep in range(N_EPISODES_PER_PREF):
    env.reset()
    nd.init_state(1)
    actor_h = critic_h = None
    v_raw, _, _, pos_a1, pos_a2 = env.step()

    for step in range(MAX_STEPS):
        v_raw_t = torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _, actor_h = actor.sample(v_raw_t, actor_h)
            feat = critic.encoder(v_raw_t).unsqueeze(1)
            lstm_out, critic_h = critic.lstm(feat, critic_h)
            lstm_out = lstm_out.squeeze(1)
            for act_vec in SAMPLE_ACTIONS:
                act_t = torch.FloatTensor([act_vec]).to(DEVICE)
                qk = critic.q(torch.cat([lstm_out, act_t], dim=-1)).item()
                record(sum_a1, cnt_a1, np.clip(pos_a1 + act_vec, -9.5, 9.5), qk)
            q_policy = critic.q(torch.cat([lstm_out, action], dim=-1))
            v_t = torch.FloatTensor(scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)
            p_mask = 0.0 if step == 0 else 1.0
            nd({'self_vision': v_t, 'self_motion': action}, q_policy / Q_SCALE, 0.0, p_mask)
            nd.detach_state()

        a_np  = action.squeeze(0).cpu().numpy()
        om_np = follower_ref.get_action(pos_a2)
        pos_a1 = np.clip(pos_a1 + a_np,   -9.5, 9.5)
        pos_a2 = np.clip(pos_a2 + om_np,  -9.5, 9.5)
        env.self_agent.p  = pos_a1
        env.other_agent.p = pos_a2
        v_raw, _, _, _, _ = env.step()

h_a1 = build_heatmap(sum_a1, cnt_a1)
print(f"  A-1: coverage={int((cnt_a1>0).sum())}/{GRID_N*GRID_N}  "
      f"min={np.nanmin(h_a1):.3f}  max={np.nanmax(h_a1):.3f}")


# =========================================================================
# A-2 Q-map（選好ごとに収集）
# =========================================================================
heatmaps_a2 = {}

for pref_name in LANDMARK_NAMES:
    print(f"\nA-2 Q-map 収集中: {pref_name} ({N_EPISODES_PER_PREF} ep)...")
    follower = LandmarkFollowerAgent(pref_name)
    sum_a2, cnt_a2 = make_acc()

    for ep in range(N_EPISODES_PER_PREF):
        env.reset()
        nd.init_state(1)
        actor_h = critic_h = None
        v_raw, _, _, pos_a1, pos_a2 = env.step()

        for step in range(MAX_STEPS):
            v_t     = torch.FloatTensor(scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)
            v_raw_t = torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action, _, actor_h = actor.sample(v_raw_t, actor_h)
                feat = critic.encoder(v_raw_t).unsqueeze(1)
                lstm_out, critic_h = critic.lstm(feat, critic_h)
                lstm_out = lstm_out.squeeze(1)
                q_policy = critic.q(torch.cat([lstm_out, action], dim=-1))

                p_mask = 0.0 if step == 0 else 1.0
                nd({'self_vision': v_t, 'self_motion': action},
                   q_policy / Q_SCALE, 0.0, p_mask)
                nd.detach_state()

                # A-2: 8 方向サンプリング
                ov_enc = nd.last_ov_enc
                for mot_vec in SAMPLE_ACTIONS:
                    mot_t = torch.FloatTensor([mot_vec]).to(DEVICE)
                    qk = net.value_estimator(ov_enc, mot_t).item()
                    record(sum_a2, cnt_a2,
                           np.clip(pos_a2 + mot_vec, -9.5, 9.5), qk)

            a_np  = action.squeeze(0).cpu().numpy()
            om_np = follower.get_action(pos_a2)
            pos_a1 = np.clip(pos_a1 + a_np,   -9.5, 9.5)
            pos_a2 = np.clip(pos_a2 + om_np,  -9.5, 9.5)
            env.self_agent.p  = pos_a1
            env.other_agent.p = pos_a2
            v_raw, _, _, _, _ = env.step()

    h = build_heatmap(sum_a2, cnt_a2)
    heatmaps_a2[pref_name] = h
    print(f"  {pref_name}: coverage={int((cnt_a2>0).sum())}/{GRID_N*GRID_N}  "
          f"min={np.nanmin(h):.3f}  max={np.nanmax(h):.3f}  mean={np.nanmean(h):.3f}")


# =========================================================================
# 描画（3×2 グリッド）
# =========================================================================
extent = [-ARENA, ARENA, -ARENA, ARENA]

def add_landmarks(ax):
    for name, lx, ly, lc in LANDMARK_INFO:
        ax.plot(lx, ly, '*', color=lc, markersize=14,
                markeredgecolor='k', markeredgewidth=0.6, label=name)

vmin_a1 = float(np.nanmin(h_a1))
vmax_a1 = float(np.nanmax(h_a1))
all_a2 = [v for v in heatmaps_a2.values()]
vmin_a2 = float(np.nanmin([np.nanmin(h) for h in all_a2]))
vmax_a2 = float(np.nanmax([np.nanmax(h) for h in all_a2]))

fig, axes = plt.subplots(3, 2, figsize=(13, 17))
fig.suptitle(
    f'Q-value Maps  [mg7: {os.path.basename(RUN_DIR)}]\n'
    f'A-2 の選好ごとに評価 — 対応ランドマーク付近が高ければ選好推論成功',
    fontsize=12)

# (0,0): A-1 参照マップ
ax = axes[0, 0]
masked = np.ma.masked_invalid(h_a1)
im = ax.imshow(masked, origin='lower', extent=extent,
               cmap='viridis', aspect='equal', vmin=vmin_a1, vmax=vmax_a1)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
add_landmarks(ax)
ax.legend(fontsize=8, loc='upper right')
ax.set_title('A-1 Q-map (8方向サンプリング, 参照)', fontsize=10)
ax.set_xlabel('X'); ax.set_ylabel('Y')

# A-2 マップ：(0,1), (1,0), (1,1), (2,0) の順
a2_slots = [(0, 1), (1, 0), (1, 1), (2, 0)]
for (row, col), pref_name in zip(a2_slots, LANDMARK_NAMES):
    lx, ly, lc = next((x, y, c) for n, x, y, c in LANDMARK_INFO if n == pref_name)
    ax = axes[row, col]
    h = heatmaps_a2[pref_name]
    masked = np.ma.masked_invalid(h)
    im = ax.imshow(masked, origin='lower', extent=extent,
                   cmap='viridis', aspect='equal', vmin=vmin_a2, vmax=vmax_a2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    add_landmarks(ax)
    ax.plot(lx, ly, 'o', color='white', markersize=18,
            markeredgecolor=lc, markeredgewidth=3, zorder=5)
    ax.set_title(f'A-2 Q-map [{pref_name} 追従評価]  ← {pref_name}({lx},{ly}) が高いはず',
                 fontsize=9)
    ax.set_xlabel('X'); ax.set_ylabel('Y')

# (2,1): 空白
axes[2, 1].axis('off')
axes[2, 1].text(0.5, 0.5,
    'A-2 の共通カラースケール\n'
    f'[vmin={vmin_a2:.2f}, vmax={vmax_a2:.2f}]\n\n'
    '白丸 = その評価での\n対象ランドマーク',
    ha='center', va='center', fontsize=10, transform=axes[2, 1].transAxes)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")
