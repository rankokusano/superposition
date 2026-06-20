"""
visualize_vision_pred_mg2.py

元論文準拠 Motion Generator 版 (MG2)：視覚予測の比較図
visualize_vision_pred_mg.py を保持したまま、新設計として独立したファイル。

visualize_vision_pred.py からの変更点:
    1. SuperpositionNetworkWithMG2 を使用
    2b. MotionGeneratorForOtherV2 (入力 = ov_enc) を使用
    2. forward の入力辞書に 'other_motion' を含めない（MG が内部生成）
    3. net.init_state(1) / net.detach_state() で Φ_s と MG を一括管理
    4. --run で run ディレクトリを指定し、出力を同ディレクトリに保存
       → 既存の vision_pred_v2.png / vision_pred_v3.png を上書きしない

可視化の構成:
    上段: A-1 側（学習目標の確認）
        Row 1: A-1 の入力視覚  v_t
        Row 2: ネットワークの予測  v̂_{t+1}   ← pred['self_vision']
        Row 3: A-1 の正解視覚  v_{t+1}
    下段: A-2 側（視点取得の確認）
        Row 4: f²_v をデコードした画像       ← Process-2 が「見ている」もの
        Row 5: A-2 の正解視野               ← シミュレーションで直接取得

    Row 2 ≈ Row 3 → A-1 の視覚予測が成功
    Row 4 ≈ Row 5 → Process-2 が A-2 の視点を獲得（視点取得）
                    ※ MG 版は om_true を使わず視覚だけで推測

使い方:
    # タイムスタンプで run ディレクトリを指定
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_vision_pred_mg2.py \\
        --run mg2_YYYYMMDD_HHMMSS

    # パス直接指定も可
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_vision_pred_mg2.py \\
        --run /work/my_research/step2/results/mg2_YYYYMMDD_HHMMSS

出力:
    {run_dir}/vision_pred_mg2.png   ← 比較図
    標準出力にピクセル誤差 (MAE) を表示
"""

import sys
import os
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '../../'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../'))
for p in [_HERE, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_green import GreenFollowerAgent
from model_mg2 import SuperpositionNetworkWithMG2
from model.modules import VisionDecoderModule

# =========================================================================
# 引数
# =========================================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    '--run', required=True,
    help='run ディレクトリのパスまたはタイムスタンプ (例: mg_20260601_201025)',
)
args = parser.parse_args()

# run ディレクトリの解決（タイムスタンプのみでも絶対パスでも動く）
if os.path.isabs(args.run):
    RUN_DIR = args.run
else:
    RUN_DIR = os.path.join(_HERE, 'results', args.run)

MODEL_FILE = os.path.join(RUN_DIR, 'model_mg2.pth')
SAVE_PATH  = os.path.join(RUN_DIR, 'vision_pred_mg2.png')

# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

DECODER_EPISODES = 20
DECODER_STEPS    = 60
DECODER_EPOCHS   = 30
N_SAMPLES        = 5
SAMPLE_STEPS     = [5, 15, 25, 35, 45]

# =========================================================================
# モデルのロード
# =========================================================================
print(f"run ディレクトリ : {RUN_DIR}")
print(f"モデル           : {MODEL_FILE}")
print(f"デバイス         : {DEVICE}")

if not os.path.exists(MODEL_FILE):
    print(f"[ERROR] {MODEL_FILE} が見つかりません。")
    sys.exit(1)

ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
Q_SCALE   = ckpt.get('q_scale',   50.0)
mg_hidden = ckpt.get('mg_hidden', 64)
print(f"  episodes={ckpt.get('episodes','?')}, "
      f"final_loss={ckpt.get('final_loss', float('nan')):.5f}, "
      f"q_scale={Q_SCALE}, mg_hidden={mg_hidden}")

model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithMG2(model_config, q_dim=1, mg_hidden=mg_hidden).to(DEVICE)
net.load_state_dict(ckpt['net_state_dict'])
net.eval()
for p in net.parameters():
    p.requires_grad = False

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
# Visual Decoder の学習（f¹_v → 入力画像の再構成）
# Process-1 の視覚特徴量から元画像を復元するデコーダーを学習し、
# 同デコーダーを Process-2 の特徴量に適用して「A-2 が見ている世界」を可視化する。
# =========================================================================
print(f"\nデコーダー学習用データ収集 ({DECODER_EPISODES} ep)...")
train_frames = []

for ep in range(DECODER_EPISODES):
    env.reset()
    v_raw, _, _, sp, op = env.step()
    self_pos  = sp.copy()
    other_pos = op.copy()
    actor_h   = None

    for _ in range(DECODER_STEPS):
        v_t = torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        train_frames.append(v_t.squeeze(0).cpu())

        with torch.no_grad():
            action, _, actor_h = actor.sample(v_t, actor_h)
        a_np  = action.squeeze(0).cpu().numpy()
        om_np = green_follower.get_action(other_pos)

        new_self  = np.clip(self_pos  + a_np,  -9.5, 9.5)
        new_other = np.clip(other_pos + om_np, -9.5, 9.5)
        env.self_agent.p  = new_self
        env.other_agent.p = new_other
        v_raw, _, _, _, _ = env.step()
        self_pos  = new_self
        other_pos = new_other

convolved_shape = net.self_vision_encoder_module.get_convolved_shape()
decoder = VisionDecoderModule(model_config.vision_decoder_module,
                              convolved_shape).to(DEVICE)
dec_opt = optim.Adam(decoder.parameters(), lr=1e-3)

frame_tensor = torch.stack(train_frames)
N = len(frame_tensor)
print(f"デコーダー学習 ({DECODER_EPOCHS} epochs, {N} frames)...")

for epoch in range(DECODER_EPOCHS):
    idx = torch.randperm(N)
    loss_sum = 0.0
    n_b = 0
    for s in range(0, N, 32):
        batch = frame_tensor[idx[s:s+32]].to(DEVICE)
        with torch.no_grad():
            f1 = net.self_vision_encoder_module(batch)
        recon = decoder(f1)
        loss  = F.l1_loss(recon, batch)
        dec_opt.zero_grad()
        loss.backward()
        dec_opt.step()
        loss_sum += loss.item(); n_b += 1
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{DECODER_EPOCHS}  loss={loss_sum/n_b:.5f}")

decoder.eval()

# =========================================================================
# サンプル収集（1エピソード時系列）
# A-1: 右下(7,-7)スタート → SAC で赤(-9,+9)へ向かう過程を記録
# A-2: 右上(7,+7)スタート → GreenFollower で緑(-9,-9)へ向かう過程を記録
# =========================================================================
TOTAL_STEPS = max(SAMPLE_STEPS) + 1

print(f"\nサンプル収集（step={SAMPLE_STEPS}）...")

start_a1 = np.array([ 7.0, -7.0])
start_a2 = np.array([ 7.0,  7.0])

env.reset()
net.init_state(1)   # Φ_s と MG を一括初期化
env.self_agent.p  = start_a1
env.other_agent.p = start_a2
v_raw, _, _, _, _ = env.step()
pos_a1 = start_a1.copy()
pos_a2 = start_a2.copy()
actor_h = critic_h = None

samples = []

for step in range(TOTAL_STEPS):
    v_t = torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        action, _, actor_h = actor.sample(v_t, actor_h)
        q_raw, critic_h   = critic(v_t, action, critic_h)

        # MG 版: other_motion は入力しない（net 内部で MG が生成）
        pred, h1, h2 = net(
            {'self_vision': v_t, 'self_motion': action},
            q_raw / Q_SCALE, 0.0, 0.0
        )
        net.detach_state()   # Φ_s と MG を一括 detach

        a_np  = action.squeeze(0).cpu().numpy()
        om_np = green_follower.get_action(pos_a2)

        new_self  = np.clip(pos_a1 + a_np,  -9.5, 9.5)
        new_other = np.clip(pos_a2 + om_np, -9.5, 9.5)
        env.self_agent.p  = new_self
        env.other_agent.p = new_other
        v_next_raw, _, _, _, _ = env.step()

    if step in SAMPLE_STEPS:
        with torch.no_grad():
            f2_v   = net.other_vision_encoder_module(v_t)
            dec_f2 = decoder(f2_v)
            v_a2_raw = env.capture_at_pos(pos_a2, pos_a1)
            v_a2 = torch.FloatTensor(
                v_a2_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        v_next_t = torch.FloatTensor(
            v_next_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        samples.append({
            'v_t':    v_t.squeeze(0).cpu(),
            'pred_v': pred['self_vision'].squeeze(0).cpu(),
            'v_next': v_next_t.squeeze(0).cpu(),
            'dec_f2': dec_f2.squeeze(0).cpu(),
            'v_a2':   v_a2.squeeze(0).cpu(),
            'pos_a1': pos_a1.copy(),
            'pos_a2': pos_a2.copy(),
            'step':   step,
        })

    v_raw = v_next_raw; pos_a1 = new_self; pos_a2 = new_other

# =========================================================================
# 可視化
# =========================================================================
def to_img(t):
    return (t.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)

fig, axes = plt.subplots(5, N_SAMPLES, figsize=(N_SAMPLES * 3.5, 5 * 2.0))
fig.patch.set_facecolor('#f8f8f8')

row_info = [
    ('#d4e8ff', "① A-1 の入力視覚  v_t",                        "入力"),
    ('#ffe0b2', "② ネットワークの予測  v̂_{t+1}",                 "予測"),
    ('#d4ffda', "③ A-1 の正解視覚  v_{t+1}",                    "正解"),
    ('#f0d4ff', "④ f²_v デコード（Process-2 の推測視点）",        "推測"),
    ('#ffd4d4', "⑤ A-2 の正解視野",                             "正解"),
]

for col, s in enumerate(samples):
    imgs = [s['v_t'], s['pred_v'], s['v_next'], s['dec_f2'], s['v_a2']]
    p1, p2 = s['pos_a1'], s['pos_a2']

    for row, (img_t, (bg, _, _)) in enumerate(zip(imgs, row_info)):
        ax = axes[row, col]
        ax.imshow(to_img(img_t))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#999'); spine.set_linewidth(0.8)

        if col == 0:
            _, long_label, _ = row_info[row]
            ax.set_ylabel(long_label, fontsize=7.5, rotation=0,
                          ha='right', va='center', labelpad=4)

        if row == 0:
            step_n = s.get('step', col)
            ax.set_title(f"step={step_n}\n"
                         f"A-1=({p1[0]:.1f},{p1[1]:.1f})\n"
                         f"A-2=({p2[0]:.1f},{p2[1]:.1f})",
                         fontsize=7)

# 仕切り線
fig.add_artist(plt.Line2D(
    [0.01, 0.99], [0.405, 0.405],
    transform=fig.transFigure,
    color='#888888', linewidth=1.5, linestyle='--'
))

fig.text(0.005, 0.62, 'A-1\n予測', fontsize=10, color='#1a5276',
         va='center', fontweight='bold', rotation=90)
fig.text(0.005, 0.20, 'A-2\n視点\n取得', fontsize=10, color='#6c3483',
         va='center', fontweight='bold', rotation=90)

run_label = os.path.basename(RUN_DIR)
fig.suptitle(
    f'視覚予測の比較  [MG2版: {run_label}]\n'
    '②≈③ならA-1の予測成功、④≈⑤なら視点取得成功\n'
    '※ MG版は other_motion を入力せず Motion Generator が内部生成',
    fontsize=10, y=0.998
)

plt.tight_layout(rect=[0.04, 0, 1, 0.97])
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

# =========================================================================
# ピクセル誤差
# =========================================================================
def mae(a, b):
    return float((a.float() - b.float()).abs().mean().item())

err_pred = np.mean([mae(s['pred_v'], s['v_next']) for s in samples])
err_dec  = np.mean([mae(s['dec_f2'], s['v_a2'])   for s in samples])
err_base = np.mean([mae(s['v_t'],    s['v_next'])  for s in samples])

print("\n=== ピクセル誤差（MAE, 0に近いほど良い）===")
print(f"  A-1 予測 (② vs ③) : {err_pred:.4f}  ← 学習の主目標")
print(f"  A-1 コピー(① vs ③): {err_base:.4f}  ← ベースライン（何も予測しない場合）")
print(f"  A-2 視点 (④ vs ⑤) : {err_dec:.4f}  ← 視点取得（MG: om 入力なし）")
print()
if err_pred < err_base:
    print("  A-1予測: ✓ ベースラインより改善（ネットワークが動きを予測できている）")
else:
    print("  A-1予測: ✗ ベースライン以下")
if err_dec < err_base:
    print("  A-2視点: ✓ ベースラインより改善（視点取得が機能している）")
else:
    print("  A-2視点: ✗ ベースライン以下（視点取得は限定的）")
