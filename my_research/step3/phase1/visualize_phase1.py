"""
visualize_phase1.py

Phase 1 訓練結果の3段階検証（Noguchi et al. Fig 4c, 4d, Fig 5 相当）。

【検証 1】h¹/h² 位置マップ（Noguchi Fig 4c 相当）
    h¹ を A-1 の (x,y) 位置でカラーリングした PCA 散布図
    h² を A-2 の (x,y) 位置でカラーリングした PCA 散布図
    カラーマップ: bilinear 混色 (BL=Blue, BR=Green, TL=Yellow, TR=Red)

【検証 2】線形回帰 R²（Noguchi Fig 4d 相当）
    h¹ → A-1 (x,y) 位置の R²
    h² → A-2 (x,y) 位置の R²
    合格基準: 両方 R² > 0.8

【検証 3】視覚復号／視点取得（Noguchi Fig 5 相当）
    Process-1 の h¹ で訓練した線形デコーダを
    Process-2 の h² に適用して A-2 視点の画像を復元できるか確認

実行コマンド（コンテナ内）:
    cd /work
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/phase1/visualize_phase1.py \\
        --run phase1_YYYYMMDD_HHMMSS \\
    2>&1 | tee my_research/step3/results/phase1_YYYYMMDD_HHMMSS/vis_log.txt

出力: {run_dir}/visualize_phase1.png
"""

import sys, os, argparse
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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
from model_phase1 import SuperpositionNetworkPhase1

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()

RUN_DIR    = (args.run if os.path.isabs(args.run)
              else os.path.join(STEP3_DIR, 'results', args.run))
MODEL_FILE = os.path.join(RUN_DIR, 'model_phase1.pth')
SAVE_PATH  = os.path.join(RUN_DIR, 'visualize_phase1.png')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

N_EPISODES = 50
MAX_STEPS  = 80
WORLD_HALF = 9.5
N_SHOW     = 4   # 視覚復号サンプル数

# =========================================================================
# モデル読み込み
# =========================================================================
print(f"run: {RUN_DIR}")
ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
Q_SCALE = ckpt.get('q_scale', 50.0)
print(f"  episodes={ckpt.get('episodes','?')}  "
      f"final_loss={ckpt.get('final_loss', float('nan')):.5f}")

model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkPhase1(model_config, q_dim=1).to(DEVICE)
net.load_state_dict(ckpt['net_state_dict'])
net.eval()
for p in net.parameters():
    p.requires_grad = False

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
# データ収集（評価時は p_mask=1.0 固定でテスト: BPTT 訓練の効果確認）
# =========================================================================
print(f"\nデータ収集中 ({N_EPISODES} ep × {MAX_STEPS} steps, eval p_mask=1.0)...")

all_h1        = []
all_h2        = []
all_self_pos  = []
all_other_pos = []
all_v_t       = []  # 全ステップの視覚（復号用）

for ep in range(N_EPISODES):
    env.reset()
    net.init_state(1)
    actor_h = critic_h = None
    v_raw, _, _, self_pos, other_pos = env.step()
    other_pos_fixed = other_pos.copy()

    for step in range(MAX_STEPS):
        v_raw_t = torch.FloatTensor(
            v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        v_t = torch.FloatTensor(
            scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action, _, actor_h = actor.sample(v_raw_t, actor_h)
            q_raw, critic_h    = critic(v_raw_t, action, critic_h)
            q_self = q_raw / Q_SCALE

            # 評価: step 0 のみ視覚あり、以降は p_mask=1.0（BPTT 訓練効果を評価）
            p_mask = 0.0 if step == 0 else 1.0
            _, h1, h2 = net(
                {'self_vision': v_t, 'self_motion': action},
                q_self, p_mask, p_mask
            )
            net.detach_state()

        all_h1.append(h1.squeeze(0).cpu().numpy())
        all_h2.append(h2.squeeze(0).cpu().numpy())
        all_self_pos.append(self_pos.copy())
        all_other_pos.append(other_pos_fixed.copy())
        all_v_t.append(v_t.squeeze(0).cpu().numpy())  # 全ステップ記録

        a_np     = action.squeeze(0).cpu().numpy()
        self_pos = np.clip(self_pos + a_np, -9.5, 9.5)
        env.self_agent.p  = self_pos
        env.other_agent.p = other_pos_fixed
        v_raw, _, _, _, _ = env.step()

H1       = np.array(all_h1)        # (N, 128)
H2       = np.array(all_h2)        # (N, 128)
SelfPos  = np.array(all_self_pos)  # (N, 2)
OtherPos = np.array(all_other_pos) # (N, 2)
V_all    = np.array(all_v_t)       # (N, 3, H, W)
N_total  = len(H1)
C, Hv, Wv = V_all.shape[1], V_all.shape[2], V_all.shape[3]
print(f"合計: {N_total} ステップ  (hidden_dim={H1.shape[1]})")


# =========================================================================
# 位置 → カラー変換 (bilinear: BL=Blue, BR=Green, TL=Yellow, TR=Red)
# =========================================================================
def pos_to_color(positions: np.ndarray) -> np.ndarray:
    nx = np.clip((positions[:, 0] + WORLD_HALF) / (2 * WORLD_HALF), 0, 1)
    ny = np.clip((positions[:, 1] + WORLD_HALF) / (2 * WORLD_HALF), 0, 1)
    r = ny
    g = nx + ny - 2 * nx * ny
    b = (1 - nx) * (1 - ny)
    return np.stack([r, g, b], axis=1)


# =========================================================================
# 検証 1: h¹/h² 位置マップ
# =========================================================================
print("\n=== 検証 1: h¹/h² 位置マップ ===")
pca1 = PCA(n_components=2); H1_2d = pca1.fit_transform(H1)
pca2 = PCA(n_components=2); H2_2d = pca2.fit_transform(H2)
var1 = pca1.explained_variance_ratio_
var2 = pca2.explained_variance_ratio_
print(f"h¹ PCA 寄与率: PC1={var1[0]:.3f}  PC2={var1[1]:.3f}  合計={var1.sum():.3f}")
print(f"h² PCA 寄与率: PC1={var2[0]:.3f}  PC2={var2[1]:.3f}  合計={var2.sum():.3f}")

colors_self  = pos_to_color(SelfPos)
colors_other = pos_to_color(OtherPos)


# =========================================================================
# 検証 2: 線形回帰 R²
# =========================================================================
print("\n=== 検証 2: 線形回帰 R² ===")

H1_tr, H1_te, SP_tr, SP_te = train_test_split(H1, SelfPos,  test_size=0.3, random_state=42)
H2_tr, H2_te, OP_tr, OP_te = train_test_split(H2, OtherPos, test_size=0.3, random_state=42)

reg_h1 = Ridge(alpha=1.0).fit(H1_tr, SP_tr)
reg_h2 = Ridge(alpha=1.0).fit(H2_tr, OP_tr)
r2_h1  = r2_score(SP_te, reg_h1.predict(H1_te))
r2_h2  = r2_score(OP_te, reg_h2.predict(H2_te))

print(f"  h¹ → A-1 位置 R² = {r2_h1:.4f}  {'[OK]' if r2_h1 > 0.8 else '[FAIL]'} (基準: >0.8)")
print(f"  h² → A-2 位置 R² = {r2_h2:.4f}  {'[OK]' if r2_h2 > 0.8 else '[FAIL]'} (基準: >0.8)")


# =========================================================================
# 検証 3: 視覚復号 / 視点取得
# (全ステップの h¹/v_t ペア 4000 サンプルで訓練)
# =========================================================================
print("\n=== 検証 3: 視覚復号 / 視点取得 ===")

V_flat = V_all.reshape(N_total, -1)  # (N, C*H*W)

idx_tr = np.random.RandomState(0).choice(N_total, int(0.7 * N_total), replace=False)
idx_te = np.setdiff1d(np.arange(N_total), idx_tr)
H1_tr2, H1_te2 = H1[idx_tr], H1[idx_te]
H2_tr2, H2_te2 = H2[idx_tr], H2[idx_te]
Vf_tr,  Vf_te  = V_flat[idx_tr], V_flat[idx_te]

dec_reg = Ridge(alpha=1.0).fit(H1_tr2, Vf_tr)
r2_dec_self  = r2_score(Vf_te, dec_reg.predict(H1_te2))
r2_dec_other = r2_score(Vf_te, dec_reg.predict(H2_te2))
print(f"  decoder R² (h¹ → 自己視覚): {r2_dec_self:.4f}")
print(f"  decoder R² (h² → 視点取得): {r2_dec_other:.4f}")

# サンプル画像（全ステップから均等にサンプル）
idx_show = np.linspace(0, len(H1_te2) - 1, N_SHOW, dtype=int)
v_orig_show   = Vf_te[idx_show].reshape(N_SHOW, C, Hv, Wv)
v_dec_h1_show = np.clip(dec_reg.predict(H1_te2[idx_show]).reshape(N_SHOW, C, Hv, Wv), -1, 1)
v_dec_h2_show = np.clip(dec_reg.predict(H2_te2[idx_show]).reshape(N_SHOW, C, Hv, Wv), -1, 1)

def to_disp(v):
    return ((np.clip(v, -1, 1) + 1) / 2).transpose(1, 2, 0)  # (H, W, C)

def make_strip(imgs):
    """(N, C, H, W) → (H, N*W, C)"""
    return np.concatenate([to_disp(imgs[i]) for i in range(len(imgs))], axis=1)


# =========================================================================
# 描画: 単一 GridSpec (height_ratios で 5 行)
# =========================================================================
fig = plt.figure(figsize=(18, 16))
gs = fig.add_gridspec(5, 3,
                       height_ratios=[3, 2.5, 0.4, 1, 1],
                       hspace=0.5, wspace=0.3)
fig.suptitle(
    f'Phase 1 Verification  [{os.path.basename(RUN_DIR)}]\n'
    f'N={N_total}  R²(h¹→A-1)={r2_h1:.3f}  R²(h²→A-2)={r2_h2:.3f}',
    fontsize=12
)

# ---- Row 0: PCA 位置マップ ----
ax = fig.add_subplot(gs[0, 0])
ax.scatter(H1_2d[:, 0], H1_2d[:, 1], c=colors_self, s=2, alpha=0.4)
ax.set_title(f'h¹  (colored by A-1 position)\nPC1={var1[0]:.2f}, PC2={var1[1]:.2f}', fontsize=9)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[0, 1])
ax.scatter(H2_2d[:, 0], H2_2d[:, 1], c=colors_other, s=2, alpha=0.4)
ax.set_title(f'h²  (colored by A-2 position)\nPC1={var2[0]:.2f}, PC2={var2[1]:.2f}', fontsize=9)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(True, alpha=0.2)

ax = fig.add_subplot(gs[0, 2])
gx, gy = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
cmap_img = np.stack([gy, gx + gy - 2*gx*gy, (1-gx)*(1-gy)], axis=2)
ax.imshow(cmap_img, origin='lower',
          extent=[-WORLD_HALF, WORLD_HALF, -WORLD_HALF, WORLD_HALF])
ax.set_title('Color reference\nBL=Blue, BR=Green, TL=Yellow, TR=Red', fontsize=9)
ax.set_xlabel('x'); ax.set_ylabel('y')

# ---- Row 1: R² 棒グラフ ----
ax = fig.add_subplot(gs[1, 0])
bars = ax.bar(['h¹→A-1 pos', 'h²→A-2 pos'], [r2_h1, r2_h2],
              color=['steelblue', 'tomato'])
ax.axhline(0.8, color='k', linestyle='--', linewidth=1.2, label='threshold 0.8')
ax.set_ylim(0, 1.05); ax.set_ylabel('R²')
ax.set_title('Linear regression R²\n(Pass: > 0.8)', fontsize=9)
ax.legend(fontsize=8)
for bar, r2 in zip(bars, [r2_h1, r2_h2]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 0.02, 1.02),
            f'{r2:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax = fig.add_subplot(gs[1, 1])
bars2 = ax.bar(['Decode h¹\n(self view)', 'Decode h²\n(perspective)'],
               [r2_dec_self, r2_dec_other], color=['steelblue', 'tomato'])
ax.set_ylim(0, 1.05); ax.set_ylabel('R²')
ax.set_title('Visual decoding R²\n(decoder trained on h¹, applied to h²)', fontsize=9)
for bar, r2 in zip(bars2, [r2_dec_self, r2_dec_other]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() + 0.02, 1.02),
            f'{r2:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_sum = fig.add_subplot(gs[1, 2])
phase1_ok = r2_h1 > 0.8 and r2_h2 > 0.8
summary = '\n'.join([
    'Phase 1 Summary',
    '',
    f'h1 PCA var: {var1.sum():.3f}',
    f'h2 PCA var: {var2.sum():.3f}',
    '',
    f'R2(h1->A-1): {r2_h1:.3f}  {"OK" if r2_h1 > 0.8 else "FAIL"}',
    f'R2(h2->A-2): {r2_h2:.3f}  {"OK" if r2_h2 > 0.8 else "FAIL"}',
    '',
    f'R2(dec h1): {r2_dec_self:.3f}',
    f'R2(dec h2): {r2_dec_other:.3f}',
    '',
    f'Phase2: {"GO" if phase1_ok else "REVISE"}',
])
ax_sum.axis('off')
ax_sum.text(0.05, 0.95, summary, transform=ax_sum.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ---- Row 2: ラベル ----
ax_lbl = fig.add_subplot(gs[2, :])
ax_lbl.axis('off')
ax_lbl.text(0.5, 0.5,
    'Visual decoding samples  '
    '[ top: original | middle: decoded from h1 | bottom: decoded from h2 (perspective-taking) ]',
    ha='center', va='center', fontsize=9, transform=ax_lbl.transAxes)

# ---- Rows 3-4: 画像ストリップ ----
for row_i, (imgs, title) in enumerate(zip(
        [v_orig_show, v_dec_h1_show, v_dec_h2_show],
        ['Original (A-1 view)',
         'Decoded from h1  (Process-1)',
         'Decoded from h2  (perspective-taking)'])):
    if row_i == 2:
        break  # row 3 と 4 に収まる2行のみ

strip = make_strip(v_orig_show)
ax = fig.add_subplot(gs[3, :])
ax.imshow(strip, aspect='auto', interpolation='nearest')
ax.set_title('Original (A-1 view)', fontsize=8, loc='left')
ax.axis('off')

strip2 = make_strip(v_dec_h1_show)
strip3 = make_strip(v_dec_h2_show)
combined = np.concatenate([strip2, strip3], axis=0)  # 2行を縦連結
ax = fig.add_subplot(gs[4, :])
ax.imshow(combined, aspect='auto', interpolation='nearest')
ax.set_title('Decoded from h1 (top) | Decoded from h2 / perspective-taking (bottom)',
             fontsize=8, loc='left')
ax.axis('off')

plt.savefig(SAVE_PATH, dpi=120, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

# =========================================================================
# サマリ
# =========================================================================
print("\n=== Phase 1 検証サマリ ===")
print(f"  [1] h¹ PCA 寄与率合計: {var1.sum():.3f}")
print(f"  [1] h² PCA 寄与率合計: {var2.sum():.3f}")
print(f"  [2] R²(h¹→A-1 pos) = {r2_h1:.4f}  {'[OK]' if r2_h1 > 0.8 else '[FAIL]'}")
print(f"  [2] R²(h²→A-2 pos) = {r2_h2:.4f}  {'[OK]' if r2_h2 > 0.8 else '[FAIL]'}")
print(f"  [3] R²(decode h¹)   = {r2_dec_self:.4f}")
print(f"  [3] R²(decode h²)   = {r2_dec_other:.4f}")
print(f"\n  Phase 1 合格: {'YES → Phase 2 へ進む' if phase1_ok else 'NO → 訓練を見直す'}")
