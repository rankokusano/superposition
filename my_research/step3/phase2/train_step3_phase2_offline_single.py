"""
train_step3_phase2_offline_single.py

単一選好（A-2=Green のみ）での Phase 2 offline 訓練。

【目的】
    診断実験: 多選好（4 択）と比較して MG r がどこまで向上するかを確認する。
    「MG が原理的に学習できるか」vs「多選好の複雑さの問題か」を切り分ける。

【既存ファイルへの影響】
    なし。新しいディレクトリ phase2_single_pref_YYYYMMDD に結果を保存。

実行コマンド（コンテナ内）:
    cd /work
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/phase2/train_step3_phase2_offline_single.py \\
    2>&1 | tee my_research/step3/results/train_phase2_single.log
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
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_EPOCHS   = 200
BATCH_SIZE   = 10
SEQ_LEN      = 100
LR           = 1e-3
WEIGHT_DECAY = 0.01
LOG_INTERVAL = 10

# 単一選好データ（多選好データは変更しない）
DATA_PATH          = os.path.join(PROJ_ROOT, 'data/data/self_random_other_stay_single_green/data.h5')
IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkMotionGenerationFeaturePrediction/default.yml')

RESUME_CHECKPOINT = None

_TIMESTAMP      = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR         = os.path.join(STEP3_DIR, 'results', f'phase2_single_pref_{_TIMESTAMP}')
MODEL_SAVE_PATH = os.path.join(RUN_DIR, 'model_phase2.pth')
LOG_PATH        = os.path.join(RUN_DIR, 'train.log')

# =========================================================================

def log(msg, f):
    print(msg); f.write(msg + '\n'); f.flush()


def scale_batch(v_raw_np):
    scaled = np.stack([scale_vision(v.transpose(2, 0, 1)) for v in v_raw_np])
    return torch.FloatTensor(scaled).to(DEVICE)


def load_dataset(h5_path, val_ratio=0.1):
    f = h5py.File(h5_path, 'r')
    sv_all = f['train']['self_vision'][:]
    sm_all = f['train']['self_motion'][:]
    f.close()
    N     = sv_all.shape[0]
    n_val = int(N * val_ratio)
    return sv_all[:-n_val], sm_all[:-n_val], sv_all[-n_val:], sm_all[-n_val:]


def train():
    os.makedirs(RUN_DIR, exist_ok=True)
    lf = open(LOG_PATH, 'w')

    log(f"実行ディレクトリ : {RUN_DIR}", lf)
    log(f"デバイス         : {DEVICE}", lf)
    log(f"データ           : {DATA_PATH}", lf)
    log(f"A-2              : Green のみ（単一選好・診断実験）", lf)
    log(f"MAX_EPOCHS       : {MAX_EPOCHS}", lf)
    log(f"BATCH_SIZE       : {BATCH_SIZE}", lf)
    log(f"SEQ_LEN          : {SEQ_LEN}", lf)
    log(f"LR               : {LR}  (AdamW, wd={WEIGHT_DECAY})", lf)

    log("\nデータ読み込み中...", lf)
    sv_train, sm_train, sv_val, sm_val = load_dataset(DATA_PATH)
    log(f"  train: {sv_train.shape[0]} ep  val: {sv_val.shape[0]} ep", lf)
    N_train, N_val = sv_train.shape[0], sv_val.shape[0]

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

    log(f"\n=== offline 訓練開始 ===\n", lf)
    best_test_loss = float('inf')

    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        idx = np.random.permutation(N_train)
        epoch_losses = []

        for b_start in range(0, N_train - BATCH_SIZE + 1, BATCH_SIZE):
            b_idx = idx[b_start: b_start + BATCH_SIZE]

            net.init_state(BATCH_SIZE)
            total_loss = torch.tensor(0.0, device=DEVICE)

            for t in range(SEQ_LEN):
                sv_t  = scale_batch(sv_train[b_idx, t])
                sm_t  = torch.FloatTensor(sm_train[b_idx, t]).to(DEVICE)
                sv_t1 = scale_batch(sv_train[b_idx, t + 1])

                p_mask = 0.0 if t == 0 else 1.0
                pred = net(
                    {'self_vision': sv_t, 'self_motion': sm_t},
                    p_mask_vision_self=p_mask,
                    p_mask_vision_other=p_mask,
                )

                state = net.superposition_module.state
                h1    = state['self'].hidden
                h2    = state['other'].hidden

                with torch.no_grad():
                    sv_enc_next = net.self_vision_encoder_module(sv_t1)
                    ov_enc_next = net.other_vision_encoder_module(sv_t1)

                L_vision     = F.l1_loss(pred['self_vision'], sv_t1)
                L_feat_self  = F.mse_loss(net.predict_feature(h1.detach()), sv_enc_next.detach())
                L_feat_other = F.mse_loss(net.predict_feature(h2), ov_enc_next.detach())
                total_loss   = total_loss + L_vision + L_feat_self + L_feat_other

            optimizer.zero_grad()
            (total_loss / SEQ_LEN).backward()
            torch.nn.utils.clip_grad_norm_(mg_params, max_norm=1.0)
            optimizer.step()
            net.detach_state()

            epoch_losses.append((total_loss / SEQ_LEN).item())

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
                        t_loss += (F.l1_loss(pred['self_vision'], sv_t1)
                                   + F.mse_loss(net.predict_feature(h2.detach()), sv_enc_next)
                                   + F.mse_loss(net.predict_feature(h2), ov_enc_next)).item()
                    test_losses.append(t_loss / SEQ_LEN)

            train_loss = np.mean(epoch_losses[-10:])
            test_loss  = np.mean(test_losses)
            mg_norm    = net.motion_generator_module.fc.weight.detach().norm().item()

            is_best = test_loss < best_test_loss
            if is_best:
                best_test_loss = test_loss
                torch.save({'net_state_dict': net.state_dict(), 'epoch': epoch,
                            'test_loss': test_loss},
                           os.path.join(RUN_DIR, 'model_best.pth'))

            log(f"Epoch {epoch:4d}/{MAX_EPOCHS} | "
                f"train={train_loss:.5f} | test={test_loss:.5f}"
                f"{'*' if is_best else ' '} | MG_L2={mg_norm:.4f}", lf)

    torch.save({'net_state_dict': net.state_dict(), 'epoch': MAX_EPOCHS,
                'data': DATA_PATH, 'phase': 2, 'pref': 'single_green'},
               MODEL_SAVE_PATH)

    log(f"\n学習完了。最終モデル: {MODEL_SAVE_PATH}", lf)
    log(f"ベストモデル: {os.path.join(RUN_DIR, 'model_best.pth')}", lf)
    log(f"best test loss: {best_test_loss:.6f}", lf)
    lf.close()


if __name__ == '__main__':
    train()
