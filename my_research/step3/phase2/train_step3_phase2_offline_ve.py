"""
train_step3_phase2_offline_ve.py

mg7 (SuperpositionNetworkWithMG3 + ValueEstimatorV2) の offline バッチ訓練版。

【Phase 2 offline との主な差分】
    モデル   : SuperpositionNetworkMotionGenerationFeaturePrediction
                → SuperpositionNetworkWithMG3 (VE 追加, q_self 追加)
    Critic   : 追加（frozen）。各ステップで q_self = Critic(sv_raw, sm) / Q_SCALE を計算。
    trainable: MG のみ → SM + MG + VE + integration + decoder + feat_pred
               ※ encoder (self/other) は frozen のまま
    LR       : 1e-3 → 1e-4 (trainable パラメータ増加に合わせて低下)
    勾配設計 : L_feat_other の h² は detach しない
               → L_feat_other → h² → SM → MG → VE へ勾配が流れる

【mg7 online との主な差分】
    環境シミュレーション不要 → h5 から offline バッチ読み込み
    バッチ訓練 (B=10) によりデータ効率向上
    Critic hidden state を各バッチ内でステップごとに更新

【データ】
    data/data/self_random_other_stay_multi_pref/data.h5
    train: 2100 ep × 101 steps  test: 220 ep × 101 steps

実行コマンド（コンテナ内）:
    cd /work
    python3 my_research/step3/phase2/train_step3_phase2_offline_ve.py \\
    2>&1 | tee my_research/step3/results/train_phase2_offline_ve.log
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
from rl_agent_sac import CriticLSTM
from model_mg3 import SuperpositionNetworkWithMG3

# =========================================================================
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_EPOCHS   = 200
BATCH_SIZE   = 10
SEQ_LEN      = 100   # t=0..99（t+1 が最終観測）
LR           = 1e-4
WEIGHT_DECAY = 0.01
LOG_INTERVAL = 10
Q_SCALE      = 50.0  # mg7 と同じ: q_self = Critic_raw / Q_SCALE
MG_HIDDEN    = 64    # mg7 と同じ: MG の hidden 次元

DATA_PATH          = os.path.join(PROJ_ROOT, 'data/data/self_random_other_stay_multi_pref/data.h5')
IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')

_TIMESTAMP      = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR         = os.path.join(STEP3_DIR, 'results', f'phase2_offline_ve_{_TIMESTAMP}')
MODEL_SAVE_PATH = os.path.join(RUN_DIR, 'model_phase2_ve.pth')
LOG_PATH        = os.path.join(RUN_DIR, 'train.log')

# =========================================================================

def log(msg, f):
    print(msg); f.write(msg + '\n'); f.flush()


def scale_batch(v_raw_np):
    """(B, H, W, C) numpy → (B, C, H, W) scale_vision 済み tensor"""
    scaled = np.stack([scale_vision(v.transpose(2, 0, 1)) for v in v_raw_np])
    return torch.FloatTensor(scaled).to(DEVICE)


def raw_batch(v_raw_np):
    """(B, H, W, C) numpy → (B, C, H, W) raw tensor（Critic 用）"""
    arr = np.ascontiguousarray(v_raw_np.transpose(0, 3, 1, 2))
    return torch.FloatTensor(arr).to(DEVICE)


def load_dataset(h5_path, val_ratio=0.1):
    f = h5py.File(h5_path, 'r')
    sv_all = f['train']['self_vision'][:]   # (N, 101, H, W, C)
    sm_all = f['train']['self_motion'][:]   # (N, 101, 2)
    f.close()
    N      = sv_all.shape[0]
    n_val  = int(N * val_ratio)
    return sv_all[:-n_val], sm_all[:-n_val], sv_all[-n_val:], sm_all[-n_val:]


def train():
    os.makedirs(RUN_DIR, exist_ok=True)
    lf = open(LOG_PATH, 'w')

    log(f"実行ディレクトリ : {RUN_DIR}", lf)
    log(f"デバイス         : {DEVICE}", lf)
    log(f"データ           : {DATA_PATH}", lf)
    log(f"モデル           : SuperpositionNetworkWithMG3 (VE 付き)", lf)
    log(f"encoder freeze   : 有効（exp1 weights, 論文準拠）", lf)
    log(f"trainable        : SM + MG + VE + integration + decoder + feat_pred", lf)
    log(f"Q_SCALE          : {Q_SCALE}", lf)
    log(f"MG_HIDDEN        : {MG_HIDDEN}", lf)
    log(f"MAX_EPOCHS       : {MAX_EPOCHS}", lf)
    log(f"BATCH_SIZE       : {BATCH_SIZE}", lf)
    log(f"SEQ_LEN          : {SEQ_LEN}", lf)
    log(f"LR               : {LR}  (AdamW, wd={WEIGHT_DECAY})", lf)

    # ── Critic (frozen) ────────────────────────────────────────────────────
    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False
    log(f"\nCritic ロード完了: {CRITIC_PATH}", lf)

    # ── SuperpositionNetworkWithMG3 ────────────────────────────────────────
    model_config = load_exp_config(IIZUKA_CONFIG_PATH)
    net = SuperpositionNetworkWithMG3(
        model_config, q_dim=1, mg_hidden=MG_HIDDEN).to(DEVICE)

    iizuka_ckpt = torch.load(IIZUKA_MODEL_PATH, map_location=DEVICE)
    net.load_iizuka_weights(iizuka_ckpt, DEVICE)
    log(f"exp1 weights ロード完了: {IIZUKA_MODEL_PATH}", lf)

    # encoder だけ freeze（SM は fresh init から訓練）
    freeze_modules = [
        net.self_vision_encoder_module,
        net.other_vision_encoder_module,
        net.share_lns,
    ]
    for module in freeze_modules:
        for p in module.parameters():
            p.requires_grad = False

    new_params = (
        list(net.superposition_module.parameters()) +
        list(net.motion_generator.parameters()) +
        list(net.value_estimator.parameters()) +
        list(net.integration_module.parameters()) +
        list(net.vision_decoder_module.parameters()) +
        list(net.feature_prediction_module.parameters())
    )
    optimizer = optim.AdamW(new_params, lr=LR, weight_decay=WEIGHT_DECAY)

    n_frozen = sum(p.numel() for m in freeze_modules for p in m.parameters())
    n_train  = sum(p.numel() for p in new_params)
    log(f"  frozen (encoder)      : {n_frozen:,}", lf)
    log(f"  trainable (SM+MG+VE+) : {n_train:,}", lf)

    # ── データ ─────────────────────────────────────────────────────────────
    log("\nデータ読み込み中...", lf)
    sv_train, sm_train, sv_val, sm_val = load_dataset(DATA_PATH, val_ratio=0.1)
    N_train, N_val = sv_train.shape[0], sv_val.shape[0]
    log(f"  train: {N_train} ep  val: {N_val} ep", lf)

    # ── 訓練ループ ─────────────────────────────────────────────────────────
    log(f"\n=== offline VE 訓練開始 ===\n", lf)
    best_test_loss = float('inf')

    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        idx = np.random.permutation(N_train)
        epoch_losses = []

        for b_start in range(0, N_train - BATCH_SIZE + 1, BATCH_SIZE):
            b_idx = idx[b_start: b_start + BATCH_SIZE]

            net.init_state(BATCH_SIZE)
            total_loss = torch.tensor(0.0, device=DEVICE)
            critic_h   = None   # Critic LSTM state をバッチ先頭でリセット

            for t in range(SEQ_LEN):
                sv_raw_t = raw_batch(sv_train[b_idx, t])         # (B, C, H, W) raw
                sv_t     = scale_batch(sv_train[b_idx, t])       # (B, C, H, W) scaled
                sm_t     = torch.FloatTensor(sm_train[b_idx, t]).to(DEVICE)  # (B, 2)
                sv_t1    = scale_batch(sv_train[b_idx, t + 1])   # (B, C, H, W) scaled

                # Critic で q_self を計算（勾配不要）
                with torch.no_grad():
                    q_raw, critic_h = critic(sv_raw_t, sm_t, critic_h)
                q_self = q_raw / Q_SCALE  # (B, 1)

                p_mask = 0.0 if t == 0 else 1.0
                pred, ss, os_ = net(
                    {'self_vision': sv_t, 'self_motion': sm_t},
                    q_self,
                    p_mask_vision_self=p_mask,
                    p_mask_vision_other=p_mask,
                )

                with torch.no_grad():
                    sv_enc_next = net.self_vision_encoder_module(sv_t1)
                    ov_enc_next = net.other_vision_encoder_module(sv_t1)

                L_vision     = F.l1_loss(pred['self_vision'], sv_t1)
                # h1 detach: L_feat_self は feature_prediction のみ更新
                L_feat_self  = F.mse_loss(net.predict_feature(ss.detach()),
                                           sv_enc_next.detach())
                # h2 detach しない: L_feat_other → h2 → SM → MG → VE へ勾配
                L_feat_other = F.mse_loss(net.predict_feature(os_),
                                           ov_enc_next.detach())
                total_loss = total_loss + L_vision + L_feat_self + L_feat_other

            optimizer.zero_grad()
            (total_loss / SEQ_LEN).backward()
            torch.nn.utils.clip_grad_norm_(new_params, max_norm=1.0)
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
                    critic_h_val = None
                    t_loss = 0.0
                    for t in range(SEQ_LEN):
                        sv_raw_t = raw_batch(sv_val[b_idx, t])
                        sv_t     = scale_batch(sv_val[b_idx, t])
                        sm_t     = torch.FloatTensor(sm_val[b_idx, t]).to(DEVICE)
                        sv_t1    = scale_batch(sv_val[b_idx, t + 1])

                        q_raw, critic_h_val = critic(sv_raw_t, sm_t, critic_h_val)
                        q_self = q_raw / Q_SCALE

                        p_mask = 0.0 if t == 0 else 1.0
                        pred, ss, os_ = net(
                            {'self_vision': sv_t, 'self_motion': sm_t},
                            q_self,
                            p_mask_vision_self=p_mask,
                            p_mask_vision_other=p_mask,
                        )
                        sv_enc_next = net.self_vision_encoder_module(sv_t1)
                        ov_enc_next = net.other_vision_encoder_module(sv_t1)

                        L_v  = F.l1_loss(pred['self_vision'], sv_t1)
                        L_fs = F.mse_loss(net.predict_feature(ss.detach()), sv_enc_next.detach())
                        L_fo = F.mse_loss(net.predict_feature(os_), ov_enc_next.detach())
                        t_loss += (L_v + L_fs + L_fo).item()
                    test_losses.append(t_loss / SEQ_LEN)

            train_loss = np.mean(epoch_losses[-10:])
            test_loss  = np.mean(test_losses)
            mg_w       = net.motion_generator.fc.weight.detach().norm().item()
            ve_w       = net.value_estimator.net[-1].weight.detach().norm().item()

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
                f"{'*' if is_best else ' '} | MG_L2={mg_w:.4f}  VE_L2={ve_w:.4f}", lf)

    # ── 最終保存 ──────────────────────────────────────────────────────────
    torch.save({
        'net_state_dict': net.state_dict(),
        'epoch': MAX_EPOCHS,
        'data': DATA_PATH,
        'model': 'SuperpositionNetworkWithMG3',
        'q_scale': Q_SCALE,
    }, MODEL_SAVE_PATH)

    log(f"\n学習完了。最終モデル: {MODEL_SAVE_PATH}", lf)
    log(f"ベストモデル: {os.path.join(RUN_DIR, 'model_best.pth')}", lf)
    log(f"best test loss: {best_test_loss:.6f}", lf)
    lf.close()


if __name__ == '__main__':
    train()
