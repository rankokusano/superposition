"""
train_step3_phase2_offline.py

Phase 2 offline バッチ訓練。
元論文 exp3 の設定（batch_size=10, 200 epoch, LR=0.001, AdamW）に準拠。

【データ】
    data/data/self_random_other_stay_multi_pref/data.h5
    train: 2100 ep × 101 steps  test: 220 ep × 101 steps

【訓練方式】
    各 batch（10 シーケンス）を t=0..99 の順に BPTT し、1 回 backward。
    t=0: p_mask=0.0（SM に実際の視覚を入力）
    t>0: p_mask=1.0（SM への視覚をマスク、MG の om のみ）

【損失】
    L_vision     : L1(pred_vision, next_vision)
    L_feat_self  : MSE(predict_feature(h¹.detach()), next_sv_enc.detach())
    L_feat_other : MSE(predict_feature(h²),           next_ov_enc.detach())
                   ↑ h² は detach しない → MG への勾配パス

実行コマンド（コンテナ内）:
    cd /work
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/phase2/train_step3_phase2_offline.py \\
    2>&1 | tee my_research/step3/results/train_phase2_offline.log
"""

import sys, os
from datetime import datetime

_HERE       = os.path.dirname(os.path.abspath(__file__))
STEP3_DIR   = os.path.abspath(os.path.join(_HERE, '..'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP3_DIR, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import h5py

from config_util import load_config as load_exp_config
from util import scale_vision
from model.model import SuperpositionNetworkMotionGenerationFeaturePrediction

# =========================================================================
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_EPOCHS  = 200
BATCH_SIZE  = 10
SEQ_LEN     = 100   # t=0..99（t+1 が最終観測）
LR          = 1e-3
WEIGHT_DECAY = 0.01
LOG_INTERVAL = 10   # epoch ごとのログ間隔

DATA_PATH   = os.path.join(PROJ_ROOT, 'data/data/self_random_other_stay_multi_pref/data.h5')
IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkMotionGenerationFeaturePrediction/default.yml')

# v3 チェックポイントから再開する場合はパスを指定、None なら飯塚から
RESUME_CHECKPOINT = None

_TIMESTAMP      = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR         = os.path.join(STEP3_DIR, 'results', f'phase2_offline_{_TIMESTAMP}')
MODEL_SAVE_PATH = os.path.join(RUN_DIR, 'model_phase2.pth')
LOG_PATH        = os.path.join(RUN_DIR, 'train.log')

# =========================================================================

def log(msg, f):
    print(msg); f.write(msg + '\n'); f.flush()


def scale_batch(v_raw_np):
    """(B, H, W, C) numpy → (B, C, H, W) scaled tensor"""
    scaled = np.stack([scale_vision(v.transpose(2, 0, 1)) for v in v_raw_np])
    return torch.FloatTensor(scaled).to(DEVICE)


def load_dataset(h5_path, val_ratio=0.1):
    """train split から val_ratio 分を検証用に分割して返す"""
    f = h5py.File(h5_path, 'r')
    sv_all = f['train']['self_vision'][:]   # (N, 101, H, W, C)
    sm_all = f['train']['self_motion'][:]   # (N, 101, 2)
    f.close()
    N      = sv_all.shape[0]
    n_val  = int(N * val_ratio)
    # 末尾を検証用（シャッフル前に固定）
    sv_tr, sm_tr = sv_all[:-n_val], sm_all[:-n_val]
    sv_va, sm_va = sv_all[-n_val:], sm_all[-n_val:]
    return sv_tr, sm_tr, sv_va, sm_va


def train():
    os.makedirs(RUN_DIR, exist_ok=True)
    lf = open(LOG_PATH, 'w')

    log(f"実行ディレクトリ : {RUN_DIR}", lf)
    log(f"デバイス         : {DEVICE}", lf)
    log(f"データ           : {DATA_PATH}", lf)
    log(f"MAX_EPOCHS       : {MAX_EPOCHS}", lf)
    log(f"BATCH_SIZE       : {BATCH_SIZE}", lf)
    log(f"SEQ_LEN          : {SEQ_LEN}", lf)
    log(f"LR               : {LR}  (AdamW, wd={WEIGHT_DECAY})", lf)

    # ── データ読み込み ─────────────────────────────────────────────────────
    log("\nデータ読み込み中...", lf)
    sv_train, sm_train, sv_val, sm_val = load_dataset(DATA_PATH, val_ratio=0.1)
    N_train = sv_train.shape[0]
    N_val   = sv_val.shape[0]
    log(f"  train: {N_train} ep  val: {N_val} ep  (total の 10% を val に使用)", lf)

    # ── モデル ─────────────────────────────────────────────────────────────
    model_config = load_exp_config(IIZUKA_CONFIG_PATH)
    net = SuperpositionNetworkMotionGenerationFeaturePrediction(model_config).to(DEVICE)

    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        ckpt = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        net.load_state_dict(ckpt.get('net_state_dict', ckpt))
        log(f"\nチェックポイントから再開: {RESUME_CHECKPOINT}", lf)
    else:
        iizuka_ckpt  = torch.load(IIZUKA_MODEL_PATH, map_location=DEVICE)
        iizuka_state = iizuka_ckpt.get('model', iizuka_ckpt)
        missing, _   = net.load_state_dict(iizuka_state, strict=False)
        log(f"\n飯塚 checkpoint 読み込み (missing={len(missing)} params)", lf)

    # MG 以外を frozen
    freeze_modules = [
        net.self_vision_encoder_module,
        net.other_vision_encoder_module,
        net.share_lns,
        net.superposition_module,
        net.integration_module,
        net.vision_decoder_module,
        net.feature_prediction_module,
    ]
    for module in freeze_modules:
        for p in module.parameters():
            p.requires_grad = False

    mg_params = list(net.motion_generator_module.parameters())
    optimizer  = optim.AdamW(mg_params, lr=LR, weight_decay=WEIGHT_DECAY)

    n_frozen = sum(p.numel() for m in freeze_modules for p in m.parameters())
    n_mg     = sum(p.numel() for p in mg_params)
    log(f"  frozen: {n_frozen:,}  trainable (MG): {n_mg:,}", lf)

    # ── 訓練ループ ─────────────────────────────────────────────────────────
    log(f"\n=== offline 訓練開始 ===\n", lf)
    best_test_loss = float('inf')

    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        idx = np.random.permutation(N_train)
        epoch_losses = []

        for b_start in range(0, N_train - BATCH_SIZE + 1, BATCH_SIZE):
            b_idx = idx[b_start: b_start + BATCH_SIZE]   # (B,)

            net.init_state(BATCH_SIZE)
            total_loss = torch.tensor(0.0, device=DEVICE)

            for t in range(SEQ_LEN):
                # 現在ステップの入力
                sv_t  = scale_batch(sv_train[b_idx, t])   # (B,C,H,W)
                sm_t  = torch.FloatTensor(sm_train[b_idx, t]).to(DEVICE)  # (B,2)

                # 次ステップの視覚（loss のターゲット）
                sv_t1 = scale_batch(sv_train[b_idx, t + 1])  # (B,C,H,W)

                p_mask = 0.0 if t == 0 else 1.0
                pred = net(
                    {'self_vision': sv_t, 'self_motion': sm_t},
                    p_mask_vision_self=p_mask,
                    p_mask_vision_other=p_mask,
                )

                state  = net.superposition_module.state
                h1     = state['self'].hidden    # (B, 128)
                h2     = state['other'].hidden   # (B, 128)

                with torch.no_grad():
                    sv_enc_next = net.self_vision_encoder_module(sv_t1)
                    ov_enc_next = net.other_vision_encoder_module(sv_t1)

                L_vision     = F.l1_loss(pred['self_vision'], sv_t1)
                L_feat_self  = F.mse_loss(net.predict_feature(h1.detach()),
                                           sv_enc_next.detach())
                L_feat_other = F.mse_loss(net.predict_feature(h2),
                                           ov_enc_next.detach())
                total_loss = total_loss + L_vision + L_feat_self + L_feat_other

            optimizer.zero_grad()
            (total_loss / SEQ_LEN).backward()
            torch.nn.utils.clip_grad_norm_(mg_params, max_norm=1.0)
            optimizer.step()
            net.detach_state()

            epoch_losses.append((total_loss / SEQ_LEN).item())

        # ── 評価 ──────────────────────────────────────────────────────────
        if epoch % LOG_INTERVAL == 0:
            net.eval()
            test_losses = []
            with torch.no_grad():
                v_idx = np.random.choice(N_val, min(100, N_val), replace=False)
                for b_start in range(0, len(v_idx) - BATCH_SIZE + 1, BATCH_SIZE):
                    b_idx = v_idx[b_start: b_start + BATCH_SIZE]
                    net.init_state(BATCH_SIZE)
                    t_loss = 0.0
                    for t in range(SEQ_LEN):
                        sv_t  = scale_batch(sv_val[b_idx, t])
                        sm_t  = torch.FloatTensor(sm_val[b_idx, t]).to(DEVICE)
                        sv_t1 = scale_batch(sv_val[b_idx, t + 1])
                        p_mask = 0.0 if t == 0 else 1.0
                        pred   = net({'self_vision': sv_t, 'self_motion': sm_t},
                                     p_mask_vision_self=p_mask,
                                     p_mask_vision_other=p_mask)
                        state  = net.superposition_module.state
                        h2     = state['other'].hidden
                        sv_enc_next = net.self_vision_encoder_module(sv_t1)
                        ov_enc_next = net.other_vision_encoder_module(sv_t1)
                        L_v  = F.l1_loss(pred['self_vision'], sv_t1)
                        L_fs = F.mse_loss(net.predict_feature(h2.detach()), sv_enc_next)
                        L_fo = F.mse_loss(net.predict_feature(h2), ov_enc_next)
                        t_loss += (L_v + L_fs + L_fo).item()
                    test_losses.append(t_loss / SEQ_LEN)

            train_loss = np.mean(epoch_losses[-10:])
            test_loss  = np.mean(test_losses)
            mg_norm    = net.motion_generator_module.fc.weight.detach().norm().item()

            is_best = test_loss < best_test_loss
            if is_best:
                best_test_loss = test_loss
                torch.save({
                    'net_state_dict': net.state_dict(),
                    'epoch': epoch,
                    'test_loss': test_loss,
                }, os.path.join(RUN_DIR, 'model_best.pth'))

            log(f"Epoch {epoch:4d}/{MAX_EPOCHS} | "
                f"train={train_loss:.5f} | test={test_loss:.5f}"
                f"{'*' if is_best else ' '} | MG_L2={mg_norm:.4f}", lf)

    # ── 最終保存 ──────────────────────────────────────────────────────────
    torch.save({
        'net_state_dict': net.state_dict(),
        'epoch': MAX_EPOCHS,
        'data': DATA_PATH,
        'phase': 2,
    }, MODEL_SAVE_PATH)

    log(f"\n学習完了。最終モデル: {MODEL_SAVE_PATH}", lf)
    log(f"ベストモデル: {os.path.join(RUN_DIR, 'model_best.pth')}", lf)
    log(f"best test loss: {best_test_loss:.6f}", lf)
    lf.close()


if __name__ == '__main__':
    train()
