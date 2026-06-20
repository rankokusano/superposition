"""
check_perspective_before_training.py

視点取得の切り分け診断。
「訓練が視点取得を壊しているのか」「アーキテクチャ変更が壊しているのか」を確認する。

比較する2つの状態:
  Left  : SuperpositionNetworkWithMG2 + exp1 重みロード直後（訓練なし）
  Right : mg4 訓練済みモデル

両者で同じ条件（静止A-2、新鮮なデコーダ）で視点取得を可視化。

期待される結果:
  Left  が Row④≈Row⑤ → 訓練が視点取得を壊している（訓練戦略の問題）
  Left  も Row④≠Row⑤ → アーキテクチャ変更が原因（Q次元追加などの問題）

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 check_perspective_before_training.py \\
        --run mg4_YYYYMMDD_HHMMSS
"""

import sys
import os
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
STEP2_DIR   = os.path.abspath(os.path.join(_HERE, '..'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP2_DIR, MY_RESEARCH, PROJ_ROOT]:
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
from util import scale_vision, de_scale_vision
from model.modules import VisionDecoderModule
from model_mg2 import SuperpositionNetworkWithMG2

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True,
                    help='mg4の run ディレクトリ (例: mg4_20260604_120000)')
args = parser.parse_args()

RUN_DIR    = (args.run if os.path.isabs(args.run)
              else os.path.join(STEP2_DIR, 'results', args.run))
MODEL_FILE = os.path.join(RUN_DIR, 'model_mg4.pth')
SAVE_PATH  = os.path.join(RUN_DIR, 'check_perspective_before_training.png')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

DECODER_EPISODES = 20
DECODER_STEPS    = 60
DECODER_EPOCHS   = 30
N_SAMPLES        = 5
SAMPLE_STEPS     = [5, 15, 25, 35, 45]

# =========================================================================
# 共通ヘルパー
# =========================================================================

def to_scaled_tensor(v_raw):
    """np.ndarray (H,W,C) → Tensor (1,C,H,W) [-1,1]"""
    return torch.FloatTensor(scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)

def to_display(t):
    """Tensor (C,H,W) [-1,1] → np.uint8 (H,W,C)"""
    img = de_scale_vision(t.cpu().numpy()).transpose(1, 2, 0).clip(0, 1)
    return (img * 255).astype(np.uint8)

def build_model_and_load(weights_path, config, q_dim=1, mg_hidden=64):
    """SuperpositionNetworkWithMG2 を作り exp1 重みをロードして返す。"""
    net = SuperpositionNetworkWithMG2(config, q_dim=q_dim, mg_hidden=mg_hidden).to(DEVICE)
    ckpt = torch.load(weights_path, map_location=DEVICE)
    net.load_iizuka_weights(ckpt, DEVICE)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net

def train_decoder(net, env, episodes, steps_per_ep, epochs):
    """
    self_vision_encoder_module の特徴量から元画像を再構成するデコーダを学習する。
    静止A-2のエピソードでフレームを収集する。
    """
    frames = []
    for _ in range(episodes):
        env.reset()
        v_raw, _, _, sp, op = env.step()
        for _ in range(steps_per_ep):
            v_t = to_scaled_tensor(v_raw)
            frames.append(v_t.squeeze(0).cpu())
            # A-1 をランダム移動
            a_np = (np.random.rand(2) - 0.5) * 0.5
            new_sp = np.clip(sp + a_np, -9.5, 9.5)
            env.self_agent.p = new_sp
            v_raw, _, _, sp, _ = env.step()

    convolved = net.self_vision_encoder_module.get_convolved_shape()
    model_config = load_exp_config(IIZUKA_CONFIG_PATH)
    decoder = VisionDecoderModule(model_config.vision_decoder_module, convolved).to(DEVICE)
    dec_opt = optim.Adam(decoder.parameters(), lr=1e-3)

    frame_t = torch.stack(frames)
    N = len(frame_t)
    for epoch in range(epochs):
        idx = torch.randperm(N)
        for s in range(0, N, 32):
            batch = frame_t[idx[s:s+32]].to(DEVICE)
            with torch.no_grad():
                f1 = net.self_vision_encoder_module(batch)
            recon = decoder(f1)
            loss = F.l1_loss(recon, batch)
            dec_opt.zero_grad(); loss.backward(); dec_opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"    decoder epoch {epoch+1}/{epochs}  loss={loss.item():.5f}")

    decoder.eval()
    return decoder

def collect_samples(net, decoder, env):
    """
    1エピソード分のサンプルを収集する。
    A-2 は静止（試行開始時のランダム位置に固定）。
    A-1 はランダム移動。
    """
    env.reset()
    net.init_state(1)
    v_raw, _, _, sp, op = env.step()
    pos_a1 = sp.copy()
    pos_a2 = op.copy()
    samples = []
    TOTAL = max(SAMPLE_STEPS) + 2

    for step in range(TOTAL):
        v_t = to_scaled_tensor(v_raw)

        with torch.no_grad():
            # LSTM を進める（Q値は 0 で渡す: 訓練前モデルの評価なので適当でよい）
            q_zero = torch.zeros(1, 1).to(DEVICE)
            a_np   = (np.random.rand(2) - 0.5) * 0.5
            pred, _, _ = net(
                {'self_vision': v_t,
                 'self_motion': torch.FloatTensor(a_np).unsqueeze(0).to(DEVICE)},
                q_zero, 0.0, 0.0
            )
            net.detach_state()

        new_sp = np.clip(pos_a1 + a_np, -9.5, 9.5)
        env.self_agent.p  = new_sp
        env.other_agent.p = pos_a2    # A-2 は動かさない
        v_next_raw, _, _, _, _ = env.step()

        if step in SAMPLE_STEPS:
            with torch.no_grad():
                f2_v   = net.other_vision_encoder_module(v_t)
                dec_f2 = decoder(f2_v)
                v_a2_raw = env.capture_at_pos(pos_a2, pos_a1)
                v_a2 = to_scaled_tensor(v_a2_raw)
                v_next_t = to_scaled_tensor(v_next_raw)

            samples.append({
                'v_t':    v_t.squeeze(0).cpu(),
                'pred_v': pred['self_vision'].squeeze(0).cpu(),
                'v_next': v_next_t.squeeze(0).cpu(),
                'dec_f2': dec_f2.squeeze(0).cpu(),
                'v_a2':   v_a2.squeeze(0).cpu(),
                'pos_a1': pos_a1.copy(),
                'pos_a2': pos_a2.copy(),
                'step': step,
            })

        v_raw  = v_next_raw
        pos_a1 = new_sp

    return samples

# =========================================================================
# 環境
# =========================================================================
model_config = load_exp_config(IIZUKA_CONFIG_PATH)
env_config   = load_config(ENV_CONFIG_PATH)
env = creator.create_environment(env_config.environment)
env.init()
env.off_display()

# =========================================================================
# モデル A: exp1 重みロード直後（訓練なし）
# =========================================================================
print("\n=== モデル A: exp1 重みロード直後（訓練なし） ===")
net_before = build_model_and_load(IIZUKA_MODEL_PATH, model_config)
print("デコーダ学習中（モデルA）...")
decoder_before = train_decoder(net_before, env, DECODER_EPISODES, DECODER_STEPS, DECODER_EPOCHS)
print("サンプル収集中（モデルA）...")
samples_before = collect_samples(net_before, decoder_before, env)

# =========================================================================
# モデル B: mg4 訓練済み
# =========================================================================
print(f"\n=== モデル B: mg4 訓練済み ({os.path.basename(RUN_DIR)}) ===")
if not os.path.exists(MODEL_FILE):
    print(f"[ERROR] {MODEL_FILE} が見つかりません。")
    sys.exit(1)

ckpt_mg4  = torch.load(MODEL_FILE, map_location=DEVICE)
mg_hidden = ckpt_mg4.get('mg_hidden', 64)
net_after = SuperpositionNetworkWithMG2(model_config, q_dim=1, mg_hidden=mg_hidden).to(DEVICE)
net_after.load_state_dict(ckpt_mg4['net_state_dict'])
net_after.eval()
for p in net_after.parameters():
    p.requires_grad = False

print("デコーダ学習中（モデルB）...")
decoder_after = train_decoder(net_after, env, DECODER_EPISODES, DECODER_STEPS, DECODER_EPOCHS)
print("サンプル収集中（モデルB）...")
samples_after = collect_samples(net_after, decoder_after, env)

# =========================================================================
# 可視化（左: 訓練前、右: 訓練後）
# =========================================================================
def mae(a, b):
    return float((a.float() - b.float()).abs().mean())

row_labels = [
    "① A-1 入力 v_t",
    "② 予測 v̂_{t+1}",
    "③ 正解 v_{t+1}",
    "④ dec(f²_v)  Process-2",
    "⑤ A-2 正解視野",
]

fig, axes = plt.subplots(5, N_SAMPLES * 2 + 1,
                         figsize=((N_SAMPLES * 2 + 1) * 2.8, 5 * 2.0),
                         gridspec_kw={'width_ratios': [1]*N_SAMPLES + [0.15] + [1]*N_SAMPLES})
fig.patch.set_facecolor('#f0f0f0')

for col in range(N_SAMPLES * 2 + 1):
    for row in range(5):
        axes[row, col].axis('off')

# 仕切り列（グレー帯）
for row in range(5):
    axes[row, N_SAMPLES].set_facecolor('#cccccc')
    axes[row, N_SAMPLES].set_visible(True)

def fill_col(samples, col_offset, color_top):
    for ci, s in enumerate(samples):
        imgs = [s['v_t'], s['pred_v'], s['v_next'], s['dec_f2'], s['v_a2']]
        col = ci + col_offset
        for row, img_t in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(to_display(img_t))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_visible(True)
            for spine in ax.spines.values():
                spine.set_edgecolor(color_top if row < 3 else '#8B0082')
                spine.set_linewidth(1.5 if row in (3, 4) else 0.8)
            if ci == 0:
                ax.set_ylabel(row_labels[row], fontsize=7, rotation=0,
                              ha='right', va='center', labelpad=4)
            if row == 0:
                p1, p2 = s['pos_a1'], s['pos_a2']
                ax.set_title(f"t={s['step']}\nA1=({p1[0]:.1f},{p1[1]:.1f})\nA2=({p2[0]:.1f},{p2[1]:.1f})",
                             fontsize=6)

fill_col(samples_before, col_offset=0,         color_top='#1a5276')
fill_col(samples_after,  col_offset=N_SAMPLES+1, color_top='#1a5276')

# 仕切り線（破線）
fig.add_artist(plt.Line2D([0.02, 0.98], [0.405, 0.405],
    transform=fig.transFigure, color='#555', linewidth=1.2, linestyle='--'))

# MAE
def print_mae(label, samples):
    ep = np.mean([mae(s['pred_v'], s['v_next']) for s in samples])
    ed = np.mean([mae(s['dec_f2'], s['v_a2'])   for s in samples])
    eb = np.mean([mae(s['v_t'],    s['v_next'])  for s in samples])
    print(f"\n{label}")
    print(f"  A-1 pred MAE   : {ep:.4f}  (baseline={eb:.4f}) → {'OK' if ep<eb else 'NG'}")
    print(f"  A-2 persp MAE  : {ed:.4f}  (baseline={eb:.4f}) → {'OK' if ed<eb else 'NG'}")
    return ep, ed, eb

ep_b, ed_b, eb_b = print_mae("モデルA（訓練前）", samples_before)
ep_a, ed_a, eb_a = print_mae("モデルB（訓練後）", samples_after)

# タイトル
fig.suptitle(
    f"視点取得 切り分け診断\n"
    f"左: exp1重みロード直後（訓練なし）  /  右: mg4訓練後\n"
    f"A-2視点MAE —  左: {ed_b:.4f}  右: {ed_a:.4f}  "
    f"(baseline={eb_b:.4f})\n"
    f"左がOKで右がNGなら「訓練が視点取得を壊している」",
    fontsize=10, y=0.999
)

plt.tight_layout(rect=[0.06, 0, 1, 0.96])
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

print("\n=== 切り分け結果 ===")
before_ok = ed_b < eb_b
after_ok  = ed_a < eb_a
if before_ok and not after_ok:
    print("→ 訓練が視点取得を壊している（訓練戦略・masking・Q値の問題）")
elif not before_ok and not after_ok:
    print("→ アーキテクチャ変更（Q次元追加など）が視点取得を壊している")
elif before_ok and after_ok:
    print("→ 訓練後も視点取得が成立している（問題なし）")
else:
    print("→ 訓練前は失敗、訓練後は成功（予想外）")
