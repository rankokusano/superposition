"""
verify_exp3.py

【目的】
    元論文 exp3 モデルを元論文のデータ（CW/CCW）で評価して MG r を確認する。
    「r≥0.87 が再現できるか」= 我々の実装の sanity check。

【比較】
    exp3 モデル + CW/CCW データ  → r=? (論文は r≥0.87)
    我々のモデル + MultiPref データ → r=0.314

実行コマンド（コンテナ内）:
    cd /work
    python3 my_research/step3/verify_exp3.py \\
    2>&1 | tee my_research/step3/results/verify_exp3.log
"""

import sys, os

_HERE       = os.path.dirname(os.path.abspath(__file__))
STEP3_DIR   = _HERE
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../..'))
for p in [_HERE, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import h5py
from scipy.stats import pearsonr

from config_util import load_config as load_exp_config
from util import scale_vision
from model.model import SuperpositionNetworkMotionGenerationFeaturePrediction

# =========================================================================
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(PROJ_ROOT, 'data/result/exp3/0/model/00200.pth')
DATA_PATH  = os.path.join(PROJ_ROOT, 'data/data/self_random_other_stay_periodic/data.h5')
CFG_PATH   = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkMotionGenerationFeaturePrediction/default.yml')

N_EP   = 200   # 評価エピソード数（test split から）
SAVE_PATH = os.path.join(STEP3_DIR, 'results', 'verify_exp3.log')

# =========================================================================

def main():
    print(f"モデル : {MODEL_PATH}")
    print(f"データ : {DATA_PATH}")
    print(f"デバイス: {DEVICE}")

    # ── モデルロード ────────────────────────────────────────────────────
    model_config = load_exp_config(CFG_PATH)
    net = SuperpositionNetworkMotionGenerationFeaturePrediction(model_config).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = ckpt.get('model', ckpt)
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    print("モデルロード完了\n")

    # ── データロード ────────────────────────────────────────────────────
    with h5py.File(DATA_PATH, 'r') as f:
        sv_all = f['test/self_vision'][:]     # (N, T, H, W, C)
        sm_all = f['test/self_motion'][:]     # (N, T, 2)
        keys = list(f['test'].keys())
        print(f"h5 test keys: {keys}")
        if 'other_motion' in keys:
            om_true_all = f['test/other_motion'][:]  # (N, T, 2)  raw agent motion
            use_pos_diff = False
            print("✓ other_motion を h5 から直接ロード")
        else:
            op_all = f['test/other_position'][:]
            om_true_all = np.diff(op_all, axis=1)   # (N, T-1, 2)  position差分
            use_pos_diff = True
            print("⚠ other_motion が h5 にない → 位置差分で代替")

    N_avail, T, H, W, C = sv_all.shape
    print(f"テストデータ: {N_avail} ep × {T} steps  (H={H}, W={W})")

    N_ep = min(N_EP, N_avail)
    sv_all     = sv_all[:N_ep]
    sm_all     = sm_all[:N_ep]
    om_true_all = om_true_all[:N_ep]

    print(f"評価エピソード数: {N_ep}")

    # ── 推論 ─────────────────────────────────────────────────────────────
    all_om_pred = []
    all_om_true = []

    # h5直接: T steps、位置差分: T-1 steps
    SEQ_LEN = om_true_all.shape[1]

    for ep in range(N_ep):
        net.init_state(1)

        for t in range(SEQ_LEN):
            sv_t = sv_all[ep, t].transpose(2, 0, 1)   # (C, H, W)
            sv_t = torch.FloatTensor(scale_vision(sv_t)).unsqueeze(0).to(DEVICE)
            sm_t = torch.FloatTensor(sm_all[ep, t]).unsqueeze(0).to(DEVICE)

            p_mask = 0.0 if t == 0 else 1.0
            with torch.no_grad():
                pred = net(
                    {'self_vision': sv_t, 'self_motion': sm_t},
                    p_mask_vision_self=p_mask,
                    p_mask_vision_other=p_mask,
                )

            om_pred_np = pred['other_motion'].squeeze(0).cpu().numpy()
            all_om_pred.append(om_pred_np)
            all_om_true.append(om_true_all[ep, t])

        if (ep + 1) % 50 == 0:
            print(f"  {ep + 1}/{N_ep} ep 処理済み")

    om_pred = np.array(all_om_pred)   # (N_total, 2)
    om_true = np.array(all_om_true)   # (N_total, 2)
    N_total = len(om_pred)
    print(f"\n合計サンプル数: {N_total}")

    # ── MG 相関 ─────────────────────────────────────────────────────────
    r_x, p_x = pearsonr(om_pred[:, 0], om_true[:, 0])
    r_y, p_y = pearsonr(om_pred[:, 1], om_true[:, 1])
    r_mean   = (r_x + r_y) / 2

    print(f"\n=== MG 相関 (Pearson r) ===")
    print(f"  r_x : {r_x:.4f}  (p={p_x:.2e})")
    print(f"  r_y : {r_y:.4f}  (p={p_y:.2e})")
    print(f"  mean: {r_mean:.4f}  {'[OK >= 0.87]' if r_mean >= 0.87 else '[FAIL < 0.87]'}")

    # ── om_true の統計（CW/CCW の運動スケール確認）──────────────────────
    src = "h5直接" if not use_pos_diff else "位置差分"
    print(f"\n=== om_true 統計 ({src}) ===")
    print(f"  |om_true| mean : {np.linalg.norm(om_true, axis=1).mean():.4f}  (PeriodicAgent v=1 なら 1.0 が期待値)")
    print(f"  |om_true| std  : {np.linalg.norm(om_true, axis=1).std():.4f}")
    print(f"  |om_pred| mean : {np.linalg.norm(om_pred, axis=1).mean():.4f}")

    # ── 比較サマリ ───────────────────────────────────────────────────────
    print(f"\n=== サマリ ===")
    print(f"  exp3 (CW/CCW, 元論文)         : MG r = {r_mean:.4f}")
    print(f"  我々 (MultiPref, 4 択)         : MG r = 0.3140")
    print(f"  我々 (SinglePref, Green のみ)  : MG r = 0.1969")
    print(f"\n  論文の閾値 r≥0.87: {'達成' if r_mean >= 0.87 else '未達成'}")


if __name__ == '__main__':
    main()
