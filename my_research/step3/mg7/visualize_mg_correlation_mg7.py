"""
visualize_mg_correlation_mg7.py

Motion Generator の出力と A-2 の実際のモーションの相関を検証する。

【目的】
    野口さんの論文では Motion Generator 出力と A-2 の実際のモーションの
    相関係数が 0.87〜0.88 と報告されている（CW/CCW パターン）。
    本スクリプトでは MultiPrefAgent（ランドマーク追従）での相関を検証し、
    Motion Generator が正しく機能していることを確認する。

【実装上の注意】
    net.forward() を呼ぶと motion_generator の LSTM 状態が内部で進む。
    別途 net.motion_generator(ov_enc) を呼ぶと2重に状態が進んでしまうため、
    forward() を手動で再現して om_generated を捕捉する。

実行コマンド（コンテナ内）:
    cd /work
    Xvfb :99 -screen 0 1024x768x24 & sleep 1
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/mg7/visualize_mg_correlation_mg7.py \\
        --run mg7_YYYYMMDD_HHMMSS \\
    2>&1 | tee my_research/step3/results/mg7_YYYYMMDD_HHMMSS/mg_corr_log.txt

出力: {run_dir}/mg_correlation_mg7.png
"""

import sys, os, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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
from model import util as model_util

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
parser.add_argument('--model_file', default='model_mg7.pth',
                    help='モデルファイル名 (default: model_mg7.pth)')
args = parser.parse_args()

RUN_DIR    = (args.run if os.path.isabs(args.run)
              else os.path.join(STEP3_DIR, 'results', args.run))
MODEL_FILE = os.path.join(RUN_DIR, args.model_file)
SAVE_PATH  = os.path.join(RUN_DIR, 'mg_correlation_mg7.png')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

N_EPISODES_PER_PREF = 25
MAX_STEPS = 80

PREF_COLORS = {'Red': 'red', 'Green': 'limegreen', 'Blue': 'blue', 'Cyan': 'cyan'}

# =========================================================================
# モデル読み込み
# =========================================================================
print(f"run: {RUN_DIR}")
ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
Q_SCALE   = ckpt.get('q_scale',   50.0)
mg_hidden = ckpt.get('mg_hidden', 64)
print(f"  episodes={ckpt.get('episodes','?')}  "
      f"final_loss={ckpt.get('final_loss', float('nan')):.5f}")
if ckpt.get('ablation_ve_disabled'):
    print("  [ablation] VE 無効モデル（評価時は VE 有効で読み込み）")

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
# データ収集
# =========================================================================
all_om_gen    = []   # MG 出力  (N, 2)
all_om_actual = []   # 実際の A-2 モーション (N, 2)
all_labels    = []

print(f"\nデータ収集中 ({N_EPISODES_PER_PREF} ep × 4 選好)...")

for pref_name in LANDMARK_NAMES:
    follower = LandmarkFollowerAgent(pref_name)

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

                # ---- forward を手動再現 (om_generated を捕捉するため) ----
                sv_enc   = net.self_vision_encoder_module(v_t)
                ov_enc   = net.other_vision_encoder_module(v_t)
                om_gen   = net.motion_generator(ov_enc)          # ← 捕捉
                q_other  = net.value_estimator(ov_enc, om_gen)

                sv_enc_m = model_util.mask(sv_enc, 0.0)
                ov_enc_m = model_util.mask(ov_enc, 0.0)

                net.superposition_module(
                    sv_enc_m, action, q_self, ov_enc_m, om_gen, q_other)
                net.detach_state()

            # A-2 の実際のモーション
            om_np = follower.get_action(other_pos)

            all_om_gen.append(om_gen.squeeze(0).cpu().numpy())
            all_om_actual.append(om_np)
            all_labels.append(pref_name)

            a_np      = action.squeeze(0).cpu().numpy()
            self_pos  = np.clip(self_pos  + a_np,  -9.5, 9.5)
            other_pos = np.clip(other_pos + om_np, -9.5, 9.5)
            env.self_agent.p  = self_pos
            env.other_agent.p = other_pos
            v_raw, _, _, _, _ = env.step()

    print(f"  {pref_name}: {N_EPISODES_PER_PREF * MAX_STEPS} steps 収集完了")

OM_gen    = np.array(all_om_gen)     # (N, 2)
OM_actual = np.array(all_om_actual)  # (N, 2)
labels    = np.array(all_labels)
N_total   = len(OM_gen)
print(f"\n合計: {N_total} ステップ")

# =========================================================================
# 相関係数
# =========================================================================
print("\n=== Motion Generator 出力と実際のモーションの相関 ===")
dims = ['x', 'y']
rs = []
for i, dim in enumerate(dims):
    r, p = pearsonr(OM_gen[:, i], OM_actual[:, i])
    rs.append(r)
    print(f"  {dim}成分: r = {r:+.4f}  (p = {p:.2e})")

print(f"\n  平均 |r| = {np.mean(np.abs(rs)):.4f}")
print(f"  参考: 野口さん論文では CW/CCW で r ≈ 0.87〜0.88")

# 選好ごとの相関
print("\n  選好ごとの相関:")
for pref in LANDMARK_NAMES:
    mask = labels == pref
    r_arr = []
    for i, dim in enumerate(dims):
        r, _ = pearsonr(OM_gen[mask, i], OM_actual[mask, i])
        r_arr.append(r)
    print(f"    {pref:6s}: rx={r_arr[0]:+.4f}  ry={r_arr[1]:+.4f}  "
          f"mean|r|={np.mean(np.abs(r_arr)):.4f}")

# =========================================================================
# 描画
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f'Motion Generator vs Actual A-2 Motion  [{os.path.basename(RUN_DIR)}]\n'
    f'({N_EPISODES_PER_PREF} ep × 4 prefs × {MAX_STEPS} steps = {N_total} points)',
    fontsize=11
)

for ax_idx, (dim, col) in enumerate(zip(['x', 'y'], [0, 1])):
    ax = axes[ax_idx]
    for pref in LANDMARK_NAMES:
        mask = labels == pref
        ax.scatter(OM_actual[mask, col], OM_gen[mask, col],
                   c=PREF_COLORS[pref], s=3, alpha=0.3, label=pref)

    # y=x の対角線（完全一致の基準線）
    lim = max(np.abs(OM_actual[:, col]).max(), np.abs(OM_gen[:, col]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1.0, alpha=0.5, label='y=x (ideal)')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)

    r_all, _ = pearsonr(OM_gen[:, col], OM_actual[:, col])
    ax.set_title(f'{dim}成分  r = {r_all:+.4f}', fontsize=11)
    ax.set_xlabel(f'Actual A-2 motion ({dim})')
    ax.set_ylabel(f'MG output ({dim})')
    ax.legend(fontsize=9, markerscale=3)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")
