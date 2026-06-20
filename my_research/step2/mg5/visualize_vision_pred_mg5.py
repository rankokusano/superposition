"""
visualize_vision_pred_mg5.py

mg5 モデル（encoder freeze 版）の視覚予測・視点取得の比較図。
mg4 版との主な違い:
    - モデルファイルが model_mg5.pth
    - Actor/Critic への入力は [0,1]（scale_vision なし）
    - encoder への入力は [-1,1]（scale_vision あり）
    - decoder も [-1,1] で学習・推論
    - check_perspective_mg5.py で切り分け診断も同時実行

使い方:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_vision_pred_mg5.py \\
        --run mg5_YYYYMMDD_HHMMSS

出力:
    {run_dir}/vision_pred_mg5.png
"""

import sys
import os
import argparse

_HERE       = os.path.dirname(os.path.abspath(__file__))
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
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_green import GreenFollowerAgent
from model_mg2 import SuperpositionNetworkWithMG2
from model.modules import VisionDecoderModule

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()

RUN_DIR    = (args.run if os.path.isabs(args.run)
              else os.path.join(STEP2_DIR, 'results', args.run))
MODEL_FILE = os.path.join(RUN_DIR, 'model_mg5.pth')
SAVE_PATH  = os.path.join(RUN_DIR, 'vision_pred_mg5.png')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
print(f"run : {RUN_DIR}")
if not os.path.exists(MODEL_FILE):
    print(f"[ERROR] {MODEL_FILE} が見つかりません"); sys.exit(1)

ckpt      = torch.load(MODEL_FILE, map_location=DEVICE)
Q_SCALE   = ckpt.get('q_scale',   50.0)
mg_hidden = ckpt.get('mg_hidden', 64)
print(f"  episodes={ckpt.get('episodes','?')}  "
      f"final_loss={ckpt.get('final_loss', float('nan')):.5f}  "
      f"encoder_frozen={ckpt.get('encoder_frozen', False)}")

model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithMG2(model_config, q_dim=1, mg_hidden=mg_hidden).to(DEVICE)
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
env.init()
env.off_display()
green_follower = GreenFollowerAgent()

# =========================================================================
# ヘルパー
# =========================================================================

def to_scaled(v_raw):
    """(H,W,C)[0,1] → Tensor(1,C,H,W)[-1,1]  encoder/decoder 用"""
    return torch.FloatTensor(scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)

def to_raw(v_raw):
    """(H,W,C)[0,1] → Tensor(1,C,H,W)[0,1]  Actor/Critic 用"""
    return torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

def to_display(t):
    """Tensor(C,H,W)[-1,1] → np.uint8(H,W,C)"""
    img = de_scale_vision(t.cpu().numpy()).transpose(1, 2, 0).clip(0, 1)
    return (img * 255).astype(np.uint8)

# =========================================================================
# decoder 学習（self_enc → 入力画像再構成、exp1 特徴が保たれているはず）
# =========================================================================
print(f"\nデコーダー学習 ({DECODER_EPISODES} ep)...")
frames = []
for ep in range(DECODER_EPISODES):
    env.reset()
    v_raw, _, _, sp, op = env.step()
    actor_h = None
    for _ in range(DECODER_STEPS):
        v_t_raw = to_raw(v_raw)
        v_t     = to_scaled(v_raw)
        frames.append(v_t.squeeze(0).cpu())
        with torch.no_grad():
            action, _, actor_h = actor.sample(v_t_raw, actor_h)
        a_np = action.squeeze(0).cpu().numpy()
        om_np = green_follower.get_action(op)
        sp = np.clip(sp + a_np, -9.5, 9.5)
        op = np.clip(op + om_np, -9.5, 9.5)
        env.self_agent.p  = sp
        env.other_agent.p = op
        v_raw, _, _, _, _ = env.step()

convolved = net.self_vision_encoder_module.get_convolved_shape()
decoder   = VisionDecoderModule(model_config.vision_decoder_module, convolved).to(DEVICE)
dec_opt   = optim.Adam(decoder.parameters(), lr=1e-3)
frame_t   = torch.stack(frames)
N         = len(frame_t)

print(f"デコーダー epochs={DECODER_EPOCHS}, frames={N}...")
for epoch in range(DECODER_EPOCHS):
    idx = torch.randperm(N)
    ls  = 0.0; nb = 0
    for s in range(0, N, 32):
        batch = frame_t[idx[s:s+32]].to(DEVICE)
        with torch.no_grad():
            f1 = net.self_vision_encoder_module(batch)
        recon = decoder(f1)
        loss  = F.l1_loss(recon, batch)
        dec_opt.zero_grad(); loss.backward(); dec_opt.step()
        ls += loss.item(); nb += 1
    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1}/{DECODER_EPOCHS}  loss={ls/nb:.5f}")
decoder.eval()

# =========================================================================
# サンプル収集
# =========================================================================
print("\nサンプル収集...")
start_a1 = np.array([ 7.0, -7.0])
start_a2 = np.array([ 7.0,  7.0])

env.reset()
net.init_state(1)
env.self_agent.p  = start_a1
env.other_agent.p = start_a2
v_raw, _, _, _, _ = env.step()
pos_a1 = start_a1.copy()
pos_a2 = start_a2.copy()
actor_h = critic_h = None
samples = []

for step in range(max(SAMPLE_STEPS) + 2):
    v_raw_t = to_raw(v_raw)
    v_t     = to_scaled(v_raw)

    with torch.no_grad():
        action, _, actor_h = actor.sample(v_raw_t, actor_h)
        q_raw, critic_h   = critic(v_raw_t, action, critic_h)
        pred, _, _ = net(
            {'self_vision': v_t, 'self_motion': action},
            q_raw / Q_SCALE, 0.0, 0.0
        )
        net.detach_state()

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
            v_a2   = to_scaled(v_a2_raw)
        v_next_t = to_scaled(v_next_raw)
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
row_info = [
    "① A-1 入力  v_t",
    "② 予測  v̂_{t+1}",
    "③ 正解  v_{t+1}",
    "④ dec(f²_v)  Process-2 推測視点",
    "⑤ A-2 正解視野",
]

fig, axes = plt.subplots(5, N_SAMPLES, figsize=(N_SAMPLES * 3.5, 5 * 2.0))
fig.patch.set_facecolor('#f8f8f8')

for col, s in enumerate(samples):
    imgs = [s['v_t'], s['pred_v'], s['v_next'], s['dec_f2'], s['v_a2']]
    p1, p2 = s['pos_a1'], s['pos_a2']
    for row, img_t in enumerate(imgs):
        ax = axes[row, col]
        ax.imshow(to_display(img_t))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#999'); sp.set_linewidth(0.8)
        if col == 0:
            ax.set_ylabel(row_info[row], fontsize=7.5, rotation=0,
                          ha='right', va='center', labelpad=4)
        if row == 0:
            ax.set_title(f"step={s['step']}\n"
                         f"A1=({p1[0]:.1f},{p1[1]:.1f})\n"
                         f"A2=({p2[0]:.1f},{p2[1]:.1f})", fontsize=7)

fig.add_artist(plt.Line2D([0.01, 0.99], [0.405, 0.405],
    transform=fig.transFigure, color='#888', linewidth=1.5, linestyle='--'))
fig.text(0.005, 0.62, 'A-1\n予測', fontsize=10, color='#1a5276',
         va='center', fontweight='bold', rotation=90)
fig.text(0.005, 0.20, 'A-2\n視点\n取得', fontsize=10, color='#6c3483',
         va='center', fontweight='bold', rotation=90)

run_label = os.path.basename(RUN_DIR)
fig.suptitle(
    f'視覚予測の比較  [MG5版: {run_label}]\n'
    '②≈③ならA-1予測成功、④≈⑤なら視点取得成功\n'
    '※ encoder freeze (exp3論文準拠)、Actor/Critic は [0,1]',
    fontsize=10, y=0.998
)
plt.tight_layout(rect=[0.04, 0, 1, 0.97])
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

def mae(a, b): return float((a.float() - b.float()).abs().mean())
err_pred = np.mean([mae(s['pred_v'], s['v_next']) for s in samples])
err_dec  = np.mean([mae(s['dec_f2'], s['v_a2'])   for s in samples])
err_base = np.mean([mae(s['v_t'],    s['v_next'])  for s in samples])

print("\n=== MAE ===")
print(f"  A-1 予測 (②vs③) : {err_pred:.4f}  baseline={err_base:.4f}  "
      f"→ {'OK' if err_pred < err_base else 'NG'}")
print(f"  A-2 視点 (④vs⑤) : {err_dec:.4f}  "
      f"→ {'OK' if err_dec < err_base else 'NG'}")
