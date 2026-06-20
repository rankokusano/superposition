"""
visualize_static_a2_exp2dec.py

mg5 encoder × exp2 学習済み decoder × 静止A-2 で視点取得を確認。
decoder 品質を exp2 レベルに固定した上で encoder が生きているかを診断する。

visualize_static_a2_mg5.py との違い:
    decoder: 新鮮学習（50 epoch）→ exp2 学習済み (data/result/exp2/0/model/00100.pth)

結果の読み方:
    ④≈⑤ → exp1 encoder の特徴は生きている。decoder の品質が問題だった
    ④≠⑤ → encoder 自体が exp2 decoder と互換でない（特徴が変化している）

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 visualize_static_a2_exp2dec.py \\
        --run mg5_YYYYMMDD_HHMMSS

出力:
    {run_dir}/vision_pred_static_a2_exp2dec.png
"""

import sys, os, argparse

_HERE       = os.path.dirname(os.path.abspath(__file__))
STEP2_DIR   = os.path.abspath(os.path.join(_HERE, '..'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP2_DIR, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from util import scale_vision, de_scale_vision
from rl_agent_sac import ActorLSTM
from model.model import Autoencoder
from model_mg2 import SuperpositionNetworkWithMG2

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()

RUN_DIR    = (args.run if os.path.isabs(args.run)
              else os.path.join(STEP2_DIR, 'results', args.run))
MODEL_FILE = os.path.join(RUN_DIR, 'model_mg5.pth')
SAVE_PATH  = os.path.join(RUN_DIR, 'vision_pred_static_a2_exp2dec.png')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
AE_CONFIG_PATH     = os.path.join(PROJ_ROOT, 'config/model/Autoencoder/default.yml')
AE_MODEL_PATH      = os.path.join(PROJ_ROOT, 'data/result/exp2/0/model/00100.pth')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

N_SAMPLES    = 5
SAMPLE_STEPS = [5, 15, 25, 35, 45]

# =========================================================================
print(f"run: {RUN_DIR}")
if not os.path.exists(MODEL_FILE):
    print(f"[ERROR] {MODEL_FILE} が見つかりません"); sys.exit(1)

# mg5 モデル（encoder freeze 済み）
ckpt = torch.load(MODEL_FILE, map_location=DEVICE)
mg_hidden = ckpt.get('mg_hidden', 64)
model_config = load_exp_config(IIZUKA_CONFIG_PATH)
net = SuperpositionNetworkWithMG2(model_config, q_dim=1, mg_hidden=mg_hidden).to(DEVICE)
net.load_state_dict(ckpt['net_state_dict'])
net.eval()
for p in net.parameters(): p.requires_grad = False
print(f"mg5 モデルロード完了  encoder_frozen={ckpt.get('encoder_frozen','?')}")

# exp2 学習済み decoder
ae_config = load_exp_config(AE_CONFIG_PATH)
ae = Autoencoder(ae_config).to(DEVICE)
ae.load_state_dict(
    torch.load(AE_MODEL_PATH, map_location=DEVICE)['model'], strict=False)
ae.eval()
for p in ae.parameters(): p.requires_grad = False
decoder = ae.ae_vision_decoder_module
print("exp2 decoder ロード完了")

actor = ActorLSTM().to(DEVICE)
actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
actor.eval()
for p in actor.parameters(): p.requires_grad = False

env_config = load_config(ENV_CONFIG_PATH)
env = creator.create_environment(env_config.environment)
env.init(); env.off_display()

# =========================================================================
def to_scaled(v_raw):
    return torch.FloatTensor(scale_vision(v_raw.transpose(2, 0, 1))).unsqueeze(0).to(DEVICE)

def to_raw(v_raw):
    return torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

def to_display(t):
    return (de_scale_vision(t.cpu().numpy()).transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)

# =========================================================================
# サンプル収集（静止A-2）
# exp1+exp2 と同条件: A-1 はランダム移動（SAC ではなく）で
# Red corner に張り付く問題を避ける
# =========================================================================
print("サンプル収集（静止A-2, A-1 ランダム移動）...")
env.reset()
net.init_state(1)
v_raw, _, _, sp, op = env.step()
pos_a1  = sp.copy()
pos_a2  = op.copy()
samples = []

print(f"  A-2 固定位置: ({pos_a2[0]:.2f}, {pos_a2[1]:.2f})")

for step in range(max(SAMPLE_STEPS) + 2):
    v_t = to_scaled(v_raw)
    with torch.no_grad():
        # A-1 はランダム移動（exp1 と同条件）
        a_np   = np.random.uniform(-1, 1, 2).astype(np.float32)
        q_zero = torch.zeros(1, 1).to(DEVICE)
        sm_t   = torch.FloatTensor(a_np).unsqueeze(0).to(DEVICE)
        pred, _, _ = net({'self_vision': v_t, 'self_motion': sm_t}, q_zero, 0.0, 0.0)
        net.detach_state()

    new_sp = np.clip(pos_a1 + a_np, -9.5, 9.5)
    env.self_agent.p  = new_sp
    env.other_agent.p = pos_a2
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

    v_raw  = v_next_raw
    pos_a1 = new_sp

# =========================================================================
# 可視化
# =========================================================================
row_labels = [
    "① A-1 入力  v_t",
    "② 予測  v̂_{t+1}",
    "③ 正解  v_{t+1}",
    "④ dec(f²_v)  [exp2 decoder]",
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
        for sp_ in ax.spines.values():
            sp_.set_edgecolor('#6c3483' if row >= 3 else '#999')
            sp_.set_linewidth(1.5 if row >= 3 else 0.8)
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=7.5, rotation=0,
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

fig.suptitle(
    f'mg5 encoder × exp2 decoder × 静止A-2  [{os.path.basename(RUN_DIR)}]\n'
    f'A-2固定=({pos_a2[0]:.1f},{pos_a2[1]:.1f})  A-1 ランダム移動\n'
    '④≈⑤ → decoder品質が問題だった  /  ④≠⑤ → encoder が exp2 と互換でない',
    fontsize=9, y=0.999
)
plt.tight_layout(rect=[0.05, 0, 1, 0.97])
plt.savefig(SAVE_PATH, dpi=150, bbox_inches='tight')
print(f"\n保存: {SAVE_PATH}")

def mae(a, b): return float((a.float() - b.float()).abs().mean())
err_dec  = np.mean([mae(s['dec_f2'], s['v_a2'])   for s in samples])
err_base = np.mean([mae(s['v_t'],    s['v_next'])  for s in samples])
print(f"\n=== MAE ===")
print(f"  A-2 視点 (④vs⑤) : {err_dec:.4f}  baseline={err_base:.4f}  → {'OK' if err_dec < err_base else 'NG'}")
if err_dec < err_base:
    print("→ decoder 品質が問題だった。exp2 decoder で解決。")
else:
    print("→ encoder 自体が exp2 decoder と互換でない。特徴が変化している。")
