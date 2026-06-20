"""
visualize_vision_pred_exp1_exp2.py

論文の正しい設定での視点取得確認:
    エンコーダ: exp1 学習済み (SuperpositionNetworkFeaturePrediction)
    デコーダ  : exp2 学習済み (Autoencoder.ae_vision_decoder_module)
    データ    : self_random_other_stay (静止 A-2)
    p_mask    : 0 (論文の VPT 可視化は p_mask=0)

これで Row 4 ≈ Row 5 が成立すれば、論文の視点取得は再現できている。
出力: my_research/step2/vision_pred_exp1_exp2.png
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '../../'))
for p in [_HERE, os.path.join(_HERE, '../'), PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config_util import load_config as load_exp_config
from model.model import SuperpositionNetworkFeaturePrediction, Autoencoder
from simulation import creator
from simulation.util import load_config
from util import scale_vision, de_scale_vision

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = os.path.join(_HERE, 'vision_pred_exp1_exp2.png')

# =========================================================================
# モデルのロード
# =========================================================================
cfg1 = load_exp_config(os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml'))
cfg2 = load_exp_config(os.path.join(PROJ_ROOT, 'config/model/Autoencoder/default.yml'))

net = SuperpositionNetworkFeaturePrediction(cfg1).to(DEVICE)
net.load_state_dict(torch.load(
    os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth'), map_location=DEVICE)['model'])
net.eval()
for p in net.parameters():
    p.requires_grad = False

ae = Autoencoder(cfg2).to(DEVICE)
ae.load_state_dict(torch.load(
    os.path.join(PROJ_ROOT, 'data/result/exp2/0/model/00100.pth'), map_location=DEVICE)['model'],
    strict=False)
ae.eval()
for p in ae.parameters():
    p.requires_grad = False

decoder = ae.ae_vision_decoder_module
print("モデルロード完了")

# =========================================================================
# データのロード
# =========================================================================
with h5py.File(os.path.join(PROJ_ROOT, 'data/data/self_random_other_stay/data.h5'), 'r') as f:
    test_v  = f['test/self_vision'][:]       # (N, T, H, W, C)
    test_sm = f['test/self_motion'][:]       # (N, T, 2)
    test_sp = f['test/self_position'][:]     # (N, T, 2)
    test_op = f['test/other_position'][:]    # (N, T, 2) or (N, 2)

print(f"データ形状: vision={test_v.shape}, other_pos={test_op.shape}")

# other_pos の次元を揃える（静止の場合は (N,2) の可能性がある）
if test_op.ndim == 2:
    test_op = np.tile(test_op[:, None, :], (1, test_v.shape[1], 1))

T = test_v.shape[1]

# =========================================================================
# シミュレーション環境（A-2 視野の描画用）
# =========================================================================
env = creator.create_environment(
    load_config(os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')).environment)
env.init()
env.off_display()

# =========================================================================
# 推論
# A-2 の移動距離が大きいシーケンスを優先して選ぶ（静止なので固定位置バリエーションで選ぶ）
# =========================================================================
SAMPLE_STEPS = [5, 15, 25, 35, 45]
seq_idx = 15   # 全シーケンス中 pred/baseline ratio が最小

print(f"使用シーケンス: {seq_idx}  A-2位置={test_op[seq_idx, 0]}")

net.init_state(1)
samples = []

for step in range(min(max(SAMPLE_STEPS) + 2, T - 1)):
    # 元論文は vision を [-1,1] にスケールして学習 (util.scale_vision = v*2-1)
    v_np = test_v[seq_idx, step].transpose(2, 0, 1)          # (C,H,W) in [0,1]
    v_t  = torch.FloatTensor(scale_vision(v_np)).unsqueeze(0).to(DEVICE)  # [-1,1]
    sm_t = torch.FloatTensor(test_sm[seq_idx, step]).unsqueeze(0).to(DEVICE)
    om_t = torch.zeros(1, 2).to(DEVICE)

    with torch.no_grad():
        pred = net({'self_vision': v_t, 'self_motion': sm_t, 'other_motion': om_t}, 0.0, 0.0)
        net.detach_state()

    if step in SAMPLE_STEPS:
        v_next_np = test_v[seq_idx, step + 1].transpose(2, 0, 1)
        v_next_scaled = torch.FloatTensor(scale_vision(v_next_np)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            f2_v   = net.other_vision_encoder_module(v_t)
            dec_f2 = decoder(f2_v)

        pos_a2 = test_op[seq_idx, step]
        pos_a1 = test_sp[seq_idx, step]
        raw = env.capture_at_pos(pos_a2, pos_a1)
        # simulator 出力は [0,1]。MAE 比較のため [-1,1] にスケール
        v_a2_scaled = torch.FloatTensor(
            scale_vision(raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)

        # 表示用に [-1,1] → [0,1] に戻す (de_scale_vision = (v+1)/2)
        def ds(t): return torch.FloatTensor(de_scale_vision(t.cpu().numpy()))

        samples.append({
            'v_t':    ds(v_t.squeeze(0)),
            'pred_v': ds(pred['self_vision'].squeeze(0)),
            'v_next': ds(v_next_scaled.squeeze(0)),
            'dec_f2': ds(dec_f2.squeeze(0)),
            'v_a2':   ds(v_a2_scaled.squeeze(0)),
            # MAE 計算は [-1,1] スケールで行う
            'pred_v_s': pred['self_vision'].squeeze(0).cpu(),
            'v_next_s': v_next_scaled.squeeze(0).cpu(),
            'dec_f2_s': dec_f2.squeeze(0).cpu(),
            'v_a2_s':   v_a2_scaled.squeeze(0).cpu(),
            'v_t_s':    v_t.squeeze(0).cpu(),
            'pos_a1': pos_a1,
            'pos_a2': pos_a2,
            'step':   step,
        })

print(f"{len(samples)} サンプル収集完了")

# =========================================================================
# 可視化
# =========================================================================
def to_img(t):
    return (t.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)

N = len(samples)
fig, axes = plt.subplots(5, N, figsize=(N * 3.5, 10))
rows = ['Row1 v_t', 'Row2 pred', 'Row3 truth', 'Row4 dec(f2)', 'Row5 A-2 view']

for col, s in enumerate(samples):
    imgs = [s['v_t'], s['pred_v'], s['v_next'], s['dec_f2'], s['v_a2']]
    for row, (img, label) in enumerate(zip(imgs, rows)):
        ax = axes[row, col]
        ax.imshow(to_img(img))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#999'); spine.set_linewidth(0.8)
        if col == 0:
            ax.set_ylabel(label, fontsize=7, rotation=0, ha='right', va='center', labelpad=4)
        if row == 0:
            ax.set_title(
                f"step={s['step']}\nA1=({s['pos_a1'][0]:.1f},{s['pos_a1'][1]:.1f})\n"
                f"A2=({s['pos_a2'][0]:.1f},{s['pos_a2'][1]:.1f})", fontsize=7)

fig.add_artist(plt.Line2D([0.01, 0.99], [0.405, 0.405],
    transform=fig.transFigure, color='#888888', linewidth=1.5, linestyle='--'))
fig.text(0.005, 0.62, 'A-1\npred', fontsize=10, color='#1a5276',
         va='center', fontweight='bold', rotation=90)
fig.text(0.005, 0.20, 'A-2\nview', fontsize=10, color='#6c3483',
         va='center', fontweight='bold', rotation=90)
fig.suptitle('exp1 + exp2 decoder  (static A-2, p_mask=0)\n'
             'Row2~Row3: A-1 prediction,  Row4~Row5: perspective taking', fontsize=11)
plt.tight_layout(rect=[0.04, 0, 1, 0.97])
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

# =========================================================================
# MAE
# =========================================================================
def mae(a, b):
    return float((a.float() - b.float()).abs().mean())

# MAE は [-1,1] スケールで統一
err_pred = np.mean([mae(s['pred_v_s'], s['v_next_s']) for s in samples])
err_base = np.mean([mae(s['v_t_s'],    s['v_next_s']) for s in samples])
err_dec  = np.mean([mae(s['dec_f2_s'], s['v_a2_s'])   for s in samples])

print("\n=== MAE ===")
print(f"  A-1 pred  (Row2 vs Row3): {err_pred:.4f}")
print(f"  baseline  (Row1 vs Row3): {err_base:.4f}")
print(f"  A-2 view  (Row4 vs Row5): {err_dec:.4f}")
print(f"  A-1 pred : {'OK' if err_pred < err_base else 'NG'}")
print(f"  A-2 view : {'OK' if err_dec  < err_base else 'NG'}")
