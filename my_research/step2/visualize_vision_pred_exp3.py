"""
visualize_vision_pred_exp3.py

元論文 exp3 (SuperpositionNetworkMotionGenerationFeaturePrediction) の
視覚予測・視点取得の比較図。mg3 版と同じ 5 行構成で結果を比較できる。

使用するもの:
    モデル : data/result/exp3/0/model/00200.pth
    データ : data/data/self_random_other_stay_periodic/data.h5
              (A-2 が CW/CCW/停止 をランダムに繰り返す)
    マスク : MotionGenerationPredictionRunner 準拠 (t=0 のみ 0, t>0 は 1.0)

可視化の構成 (mg3 版と同じ):
    Row 1: A-1 の入力視覚  v_t
    Row 2: ネットワークの予測  v̂_{t+1}
    Row 3: A-1 の正解視覚  v_{t+1}
    Row 4: decoder(f²_v)  ← Process-2 の推測視点
    Row 5: A-2 の正解視野

出力:
    my_research/step2/vision_pred_exp3.png
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '../../'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../'))
for p in [_HERE, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from model.model import SuperpositionNetworkMotionGenerationFeaturePrediction, Autoencoder
from model.modules import VisionDecoderModule
from util import scale_vision, de_scale_vision

# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH     = os.path.join(PROJ_ROOT, 'data/result/exp3/0/model/00200.pth')
AE_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp2/0/model/00100.pth')
DATA_PATH      = os.path.join(PROJ_ROOT, 'data/data/self_random_other_stay_periodic/data.h5')
MODEL_CFG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkMotionGenerationFeaturePrediction/default.yml')
AE_CFG_PATH    = os.path.join(PROJ_ROOT, 'config/model/Autoencoder/default.yml')
ENV_CFG_PATH   = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')
SAVE_PATH      = os.path.join(_HERE, 'vision_pred_exp3.png')

N_SAMPLES       = 5
SAMPLE_STEPS    = [5, 15, 25, 35, 45]

# =========================================================================
print(f"モデル  : {MODEL_PATH}")
print(f"データ  : {DATA_PATH}")
print(f"デバイス: {DEVICE}")

model_config = load_exp_config(MODEL_CFG_PATH)
net = SuperpositionNetworkMotionGenerationFeaturePrediction(model_config).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = ckpt.get('model', ckpt)
net.load_state_dict(state_dict, strict=False)
net.eval()
for p in net.parameters():
    p.requires_grad = False
print("exp3 モデルロード完了")

# =========================================================================
# exp2 Autoencoder のデコーダをロード（論文で使われている学習済みデコーダ）
# exp2 はエンコーダを exp1 から転送して固定し、ae_vision_decoder_module だけ学習
# =========================================================================
ae_config = load_exp_config(AE_CFG_PATH)
ae = Autoencoder(ae_config).to(DEVICE)
ae_ckpt = torch.load(AE_MODEL_PATH, map_location=DEVICE)
ae.load_state_dict(ae_ckpt['model'], strict=False)
ae.eval()
for p in ae.parameters():
    p.requires_grad = False
decoder = ae.ae_vision_decoder_module
print("exp2 Autoencoder デコーダロード完了")

# =========================================================================
# テストシーケンスを h5 から読み込んで推論
# A-2 の正解視野はシミュレーションで描画
# =========================================================================
print(f"\nテストシーケンス推論...")

with h5py.File(DATA_PATH, 'r') as f:
    test_visions   = f['test/self_vision'][:]     # (N_test, T, H, W, C)
    test_motions   = f['test/self_motion'][:]     # (N_test, T, 2)
    test_self_pos  = f['test/self_position'][:]   # (N_test, T, 2)
    test_other_pos = f['test/other_position'][:]  # (N_test, T, 2)

# シミュレーション環境 (A-2 視野の描画用)
env_config = load_config(ENV_CFG_PATH)
env = creator.create_environment(env_config.environment)
env.init()
env.off_display()

# A-2 の動きが大きいシーケンスを選ぶ
other_disp = np.linalg.norm(
    test_other_pos[:, -1] - test_other_pos[:, 0], axis=-1)
seq_idx = int(np.argsort(other_disp)[::-1][0])   # 最も A-2 が動いたシーケンス

print(f"使用シーケンス: {seq_idx}  A-2移動距離={other_disp[seq_idx]:.2f}")

sv_seq  = test_visions[seq_idx]    # (T, H, W, C)
sm_seq  = test_motions[seq_idx]    # (T, 2)
sp_seq  = test_self_pos[seq_idx]   # (T, 2)
op_seq  = test_other_pos[seq_idx]  # (T, 2)

TOTAL_STEPS = max(SAMPLE_STEPS) + 1

net.init_state(1)
samples = []

def ds(t): return torch.FloatTensor(de_scale_vision(t.cpu().numpy()))

for step in range(TOTAL_STEPS):
    # exp3 モデルは [-1,1] スケールで学習
    v_np = sv_seq[step].transpose(2, 0, 1)
    v_t  = torch.FloatTensor(scale_vision(v_np)).unsqueeze(0).to(DEVICE)
    sm_t = torch.FloatTensor(sm_seq[step]).unsqueeze(0).to(DEVICE)

    p_mask_other = 0.0 if step == 0 else 1.0

    with torch.no_grad():
        pred = net(
            {'self_vision': v_t, 'self_motion': sm_t},
            p_mask_vision_self=0.0,
            p_mask_vision_other=p_mask_other,
        )
        net.detach_state()

    if step in SAMPLE_STEPS:
        v_next_np = sv_seq[step + 1].transpose(2, 0, 1)
        v_next_s  = torch.FloatTensor(scale_vision(v_next_np)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            f2_v   = net.other_vision_encoder_module(v_t)
            dec_f2 = decoder(f2_v)
            v_a2_raw = env.capture_at_pos(op_seq[step], sp_seq[step])
            v_a2_s = torch.FloatTensor(
                scale_vision(v_a2_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)
        samples.append({
            # 表示用: [-1,1] → [0,1]
            'v_t':    ds(v_t.squeeze(0)),
            'pred_v': ds(pred['self_vision'].squeeze(0)),
            'v_next': ds(v_next_s.squeeze(0)),
            'dec_f2': ds(dec_f2.squeeze(0)),
            'v_a2':   ds(v_a2_s.squeeze(0)),
            # MAE 用: [-1,1] スケール
            'pred_v_s': pred['self_vision'].squeeze(0).cpu(),
            'v_next_s': v_next_s.squeeze(0).cpu(),
            'dec_f2_s': dec_f2.squeeze(0).cpu(),
            'v_a2_s':   v_a2_s.squeeze(0).cpu(),
            'v_t_s':    v_t.squeeze(0).cpu(),
            'pos_a1': sp_seq[step].copy(),
            'pos_a2': op_seq[step].copy(),
            'step':   step,
        })

# =========================================================================
# 可視化
# =========================================================================
def to_img(t):
    return (t.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)

fig, axes = plt.subplots(5, N_SAMPLES, figsize=(N_SAMPLES * 3.5, 5 * 2.0))
fig.patch.set_facecolor('#f8f8f8')

row_info = [
    ('#d4e8ff', "Row1 A-1 input v_t"),
    ('#ffe0b2', "Row2 prediction v_{t+1}"),
    ('#d4ffda', "Row3 A-1 truth v_{t+1}"),
    ('#f0d4ff', "Row4 decoder(f2_v) Process-2 view"),
    ('#ffd4d4', "Row5 A-2 true view"),
]

for col, s in enumerate(samples):
    imgs = [s['v_t'], s['pred_v'], s['v_next'], s['dec_f2'], s['v_a2']]
    p1, p2 = s['pos_a1'], s['pos_a2']
    for row, (img_t, (bg, label)) in enumerate(zip(imgs, row_info)):
        ax = axes[row, col]
        ax.imshow(to_img(img_t))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#999'); spine.set_linewidth(0.8)
        if col == 0:
            ax.set_ylabel(label, fontsize=7, rotation=0,
                          ha='right', va='center', labelpad=4)
        if row == 0:
            ax.set_title(f"step={s['step']}\n"
                         f"A1=({p1[0]:.1f},{p1[1]:.1f})\n"
                         f"A2=({p2[0]:.1f},{p2[1]:.1f})", fontsize=7)

fig.add_artist(plt.Line2D(
    [0.01, 0.99], [0.405, 0.405],
    transform=fig.transFigure,
    color='#888888', linewidth=1.5, linestyle='--'))

fig.text(0.005, 0.62, 'A-1\npred', fontsize=10, color='#1a5276',
         va='center', fontweight='bold', rotation=90)
fig.text(0.005, 0.20, 'A-2\nview', fontsize=10, color='#6c3483',
         va='center', fontweight='bold', rotation=90)

fig.suptitle('exp3 (original paper, moving A-2, MG+mask)\n'
             'Row2~Row3: A-1 prediction,  Row4~Row5: perspective taking',
             fontsize=10, y=0.998)

plt.tight_layout(rect=[0.04, 0, 1, 0.97])
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

# =========================================================================
# MAE
# =========================================================================
def mae(a, b):
    return float((a.float() - b.float()).abs().mean())

err_pred = np.mean([mae(s['pred_v_s'], s['v_next_s']) for s in samples])
err_dec  = np.mean([mae(s['dec_f2_s'], s['v_a2_s'])  for s in samples])
err_base = np.mean([mae(s['v_t_s'],    s['v_next_s']) for s in samples])

print("\n=== MAE ===")
print(f"  A-1 pred  (Row2 vs Row3): {err_pred:.4f}")
print(f"  baseline  (Row1 vs Row3): {err_base:.4f}")
print(f"  A-2 view  (Row4 vs Row5): {err_dec:.4f}")
print()
print(f"  A-1 pred : {'OK' if err_pred < err_base else 'NG'}")
print(f"  A-2 view : {'OK' if err_dec  < err_base else 'NG'}")
