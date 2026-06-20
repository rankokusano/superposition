"""
train_step3_phase1.py

Phase 1: Superposition Module ベーストレーニング（A-2 静止条件）。

【目的】
    Noguchi et al. の二段階訓練の第一段階を実装する。
    A-2 が静止した状態で SM が「自己/他者の位置を h¹/h² で表現できる」能力を獲得する。
    Phase 2 では本 Phase で学習した SM を frozen にして MG のみ学習する。

【設計方針】
    - A-2 は静止: other_pos は変化しない（om_np = zeros(2)）
    - 両視覚エンコーダを frozen（飯塚モデルから転送）
    - step 0 のみ両視覚マスクなし、以降は両方 1.0 でマスク
    - 訓練対象: SM, integration, decoder, feature_prediction
    - 損失: L_vision + L_feat_self + L_feat_other（mg7 と同じ）

【検証】
    訓練後に visualize_phase1.py で以下を確認:
        1. h¹ PCA (A-1 位置で色付け) + h² PCA (A-2 位置で色付け)
        2. 線形回帰 R² (h¹→A-1 位置 > 0.8, h²→A-2 位置 > 0.8)
        3. 視覚復号: Process-1 で学習した decoder を Process-2 に適用

保存先: step3/results/phase1_YYYYMMDD_HHMMSS/

実行コマンド（コンテナ内）:
    cd /work
    Xvfb :99 -screen 0 1024x768x24 & sleep 1
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/phase1/train_step3_phase1.py \\
    2>&1 | tee my_research/step3/results/train_phase1.log
"""

import sys
import os
from datetime import datetime

_HERE       = os.path.dirname(os.path.abspath(__file__))
STEP3_DIR   = os.path.abspath(os.path.join(_HERE, '..'))
STEP2_DIR   = os.path.abspath(os.path.join(_HERE, '../../step2'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP3_DIR, STEP2_DIR, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from util import scale_vision

from rl_agent_sac import ActorLSTM, CriticLSTM
from model_phase1 import SuperpositionNetworkPhase1

# =========================================================================
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES  = 1000
MAX_STEPS = 100
Q_SCALE   = 50.0
LR_NEW    = 1e-4

IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

LOG_INTERVAL    = 50
_TIMESTAMP      = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR         = os.path.join(STEP3_DIR, 'results', f'phase1_{_TIMESTAMP}')
MODEL_SAVE_PATH = os.path.join(RUN_DIR, 'model_phase1.pth')
LOG_PATH        = os.path.join(RUN_DIR, 'train.log')

# =========================================================================

def log(msg, f):
    print(msg); f.write(msg + '\n'); f.flush()

def preprocess_vision(v_raw):
    v = scale_vision(v_raw.transpose(2, 0, 1))
    return torch.FloatTensor(v).unsqueeze(0).to(DEVICE)

def preprocess_vision_raw(v_raw):
    return torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

def to_tensor(arr):
    return torch.FloatTensor(arr).unsqueeze(0).to(DEVICE)

# =========================================================================

def train():
    os.makedirs(RUN_DIR, exist_ok=True)
    lf = open(LOG_PATH, 'w')

    log(f"実行ディレクトリ : {RUN_DIR}", lf)
    log(f"デバイス         : {DEVICE}", lf)
    log(f"Phase            : 1 (SM ベース訓練、A-2 静止)", lf)
    log(f"encoder freeze   : 有効（飯塚モデルから転送）", lf)
    log(f"A-2 行動         : 静止 (om = zeros(2))", lf)
    log(f"マスク戦略       : step==0 のみ p_mask=0.0、以降 p_mask=0.99 (元論文 exp1 準拠)", lf)
    log(f"Q_SCALE          : {Q_SCALE}", lf)
    log(f"LR_NEW           : {LR_NEW}", lf)
    log(f"EPISODES         : {EPISODES}", lf)
    log(f"MAX_STEPS        : {MAX_STEPS}", lf)

    # ── Actor / Critic (固定) ─────────────────────────────────────────────
    actor = ActorLSTM().to(DEVICE)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
    actor.eval()
    for p in actor.parameters():
        p.requires_grad = False

    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
    critic.eval()
    for p in critic.parameters():
        p.requires_grad = False

    # ── SuperpositionNetworkPhase1 ────────────────────────────────────────
    model_config = load_exp_config(IIZUKA_CONFIG_PATH)
    net = SuperpositionNetworkPhase1(model_config, q_dim=1).to(DEVICE)

    iizuka_ckpt = torch.load(IIZUKA_MODEL_PATH, map_location=DEVICE)
    net.load_iizuka_weights(iizuka_ckpt, DEVICE)

    for module in [net.self_vision_encoder_module,
                   net.other_vision_encoder_module,
                   net.share_lns]:
        for p in module.parameters():
            p.requires_grad = False

    new_params = (
        list(net.superposition_module.parameters()) +
        list(net.integration_module.parameters()) +
        list(net.vision_decoder_module.parameters()) +
        list(net.feature_prediction_module.parameters())
    )
    optimizer = optim.Adam(new_params, lr=LR_NEW)

    n_frozen = sum(p.numel() for m in [net.self_vision_encoder_module,
                                        net.other_vision_encoder_module,
                                        net.share_lns]
                   for p in m.parameters())
    n_train  = sum(p.numel() for p in new_params)
    log(f"\nパラメータ数:", lf)
    log(f"  frozen (encoder)  : {n_frozen:,}", lf)
    log(f"  trainable (SM 等) : {n_train:,}", lf)

    # ── 環境 ─────────────────────────────────────────────────────────────
    env_config = load_config(ENV_CONFIG_PATH)
    env = creator.create_environment(env_config.environment)
    env.init()
    env.off_display()

    log(f"学習方式         : BPTT (エピソード全体で loss 累積、1回 backward)", lf)
    log(f"\n=== Phase 1 学習開始 ===\n", lf)

    episode_losses = []

    for episode in range(EPISODES):
        env.reset()
        net.init_state(1)
        net.train()

        actor_h = critic_h = None
        v_raw, _, _, self_pos, other_pos = env.step()
        other_pos_fixed = other_pos.copy()  # A-2 は静止

        ep_total_loss = torch.tensor(0.0, device=DEVICE)

        for step in range(MAX_STEPS):
            v_raw_t = preprocess_vision_raw(v_raw)
            with torch.no_grad():
                action, _, actor_h = actor.sample(v_raw_t, actor_h)
                q_raw, critic_h    = critic(v_raw_t, action, critic_h)

            v_t    = preprocess_vision(v_raw)
            q_self = q_raw / Q_SCALE
            a_np   = action.squeeze(0).cpu().numpy()

            # step 0 のみ視覚あり、以降は確率 0.99 でマスク（元論文 exp1 準拠）
            p_mask = 0.0 if step == 0 else 0.99
            pred, ss, os_ = net(
                {'self_vision': v_t, 'self_motion': to_tensor(a_np)},
                q_self, p_mask, p_mask
            )

            new_self = np.clip(self_pos + a_np, -9.5, 9.5)
            env.self_agent.p  = new_self
            env.other_agent.p = other_pos_fixed
            v_next_raw, _, _, _, _ = env.step()

            v_next_t = preprocess_vision(v_next_raw)
            with torch.no_grad():
                sv_enc_next = net.self_vision_encoder_module(v_next_t)
                ov_enc_next = net.other_vision_encoder_module(v_next_t)

            loss_vision     = F.l1_loss(pred['self_vision'], v_next_t)
            # feature pred は predict_feature の訓練のみ（LSTM には勾配を流さない）
            pred_self_feat  = net.predict_feature(ss.detach())
            pred_other_feat = net.predict_feature(os_.detach())
            loss_feat_self  = F.mse_loss(pred_self_feat,  sv_enc_next.detach())
            loss_feat_other = F.mse_loss(pred_other_feat, ov_enc_next.detach())
            step_loss = loss_vision + loss_feat_self + loss_feat_other

            # BPTT: detach せずに loss を積算（LSTM 状態を計算グラフに保持）
            ep_total_loss = ep_total_loss + step_loss

            v_raw    = v_next_raw
            self_pos = new_self

        # エピソード終了後に一括更新
        optimizer.zero_grad()
        (ep_total_loss / MAX_STEPS).backward()
        torch.nn.utils.clip_grad_norm_(new_params, max_norm=1.0)
        optimizer.step()

        # 次エピソードのために状態を切り離す
        net.detach_state()

        ep_loss = (ep_total_loss / MAX_STEPS).item()
        episode_losses.append(ep_loss)

        if (episode + 1) % LOG_INTERVAL == 0:
            recent  = episode_losses[-LOG_INTERVAL:]
            sp_norm = net.superposition_module.lstm.weight_ih.detach().norm().item()
            log(f"Episode {episode+1:5d}/{EPISODES} | "
                f"Loss={np.mean(recent):.5f} | "
                f"Φs_L2={sp_norm:.4f}", lf)

    torch.save({
        'net_state_dict':       net.state_dict(),
        'superposition_module': net.superposition_module.state_dict(),
        'encoder_frozen':       True,
        'scale_vision':         True,
        'a2_agent':             'stationary',
        'p_mask':               'step>0 → 0.99 stochastic (exp1 準拠)',
        'training':             'BPTT (episode-level update)',
        'q_scale':              Q_SCALE,
        'lr_new':               LR_NEW,
        'episodes':             EPISODES,
        'max_steps':            MAX_STEPS,
        'final_loss':           float(np.mean(episode_losses[-LOG_INTERVAL:])),
        'phase':                1,
    }, MODEL_SAVE_PATH)

    log(f"\n学習完了。保存: {MODEL_SAVE_PATH}", lf)
    log(f"最終平均Loss : {np.mean(episode_losses[-LOG_INTERVAL:]):.6f}", lf)
    lf.close()


if __name__ == '__main__':
    train()
