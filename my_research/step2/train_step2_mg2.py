"""
train_step2_mg2.py

元論文アーキテクチャに即した Step2 学習スクリプト。
train_step2_mg.py からの変更点:

    SuperpositionNetworkWithMG  → SuperpositionNetworkWithMG2
    motion_generator 入力: h²_{t-1} → ov_enc  (元論文準拠)
    保存先: results/mg2_YYYYMMDD_HHMMSS/

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 train_step2_mg2.py
"""

import sys
import os
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '../../'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../'))
for p in [_HERE, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config

from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_green import GreenFollowerAgent
from model_mg2 import SuperpositionNetworkWithMG2

# =========================================================================
# ハイパーパラメータ
# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES  = 1000
MAX_STEPS = 100
Q_SCALE   = 50.0
MG_HIDDEN = 64

LR_ENCODER = 1e-5
LR_NEW     = 1e-4

IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

LOG_INTERVAL    = 50
_TIMESTAMP      = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR         = os.path.join(_HERE, 'results', f'mg2_{_TIMESTAMP}')
MODEL_SAVE_PATH = os.path.join(RUN_DIR, 'model_mg2.pth')
LOG_PATH        = os.path.join(RUN_DIR, 'train.log')

# =========================================================================
# ユーティリティ
# =========================================================================

def log(msg: str, log_file):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

def preprocess_vision(v_raw):
    return torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

def to_motion_tensor(m_np):
    return torch.FloatTensor(m_np).unsqueeze(0).to(DEVICE)

def get_diagnostics(net):
    mg_fc_norm = net.motion_generator.fc.weight.detach().norm().item()
    sp_ih_norm = net.superposition_module.lstm.weight_ih.detach().norm().item()
    return mg_fc_norm, sp_ih_norm

# =========================================================================
# encoder 分化の定量チェック
# =========================================================================

def check_encoder_similarity(net, n=100):
    """self_enc と other_enc の cosine 類似度を計算して返す。"""
    sims = []
    with torch.no_grad():
        for _ in range(n):
            v = torch.rand(1, 3, 16, 64).to(DEVICE)
            f1 = net.self_vision_encoder_module(v)
            f2 = net.other_vision_encoder_module(v)
            cos = F.cosine_similarity(f1, f2).item()
            sims.append(cos)
    return float(np.mean(sims)), float(np.std(sims))

# =========================================================================
# 学習ループ
# =========================================================================

def train():
    os.makedirs(RUN_DIR, exist_ok=True)
    log_file = open(LOG_PATH, 'w')

    log(f"実行ディレクトリ : {RUN_DIR}", log_file)
    log(f"デバイス         : {DEVICE}", log_file)
    log(f"MG 入力          : ov_enc (元論文準拠)", log_file)
    log(f"Q_SCALE          : {Q_SCALE}", log_file)
    log(f"MG_HIDDEN        : {MG_HIDDEN}", log_file)
    log(f"LR_ENCODER       : {LR_ENCODER}", log_file)
    log(f"LR_NEW           : {LR_NEW}", log_file)
    log(f"EPISODES         : {EPISODES}", log_file)
    log(f"保存先           : {MODEL_SAVE_PATH}", log_file)

    # ── A-1 Actor / Critic (固定) ──────────────────────────────────────────
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

    # ── SuperpositionNetworkWithMG2 の初期化 ──────────────────────────────
    model_config = load_exp_config(IIZUKA_CONFIG_PATH)
    net = SuperpositionNetworkWithMG2(model_config, q_dim=1, mg_hidden=MG_HIDDEN).to(DEVICE)

    iizuka_ckpt = torch.load(IIZUKA_MODEL_PATH, map_location=DEVICE)
    net.load_iizuka_weights(iizuka_ckpt, DEVICE)

    # ── 差分学習率 Optimizer ───────────────────────────────────────────────
    encoder_params = (
        list(net.self_vision_encoder_module.parameters()) +
        list(net.other_vision_encoder_module.parameters()) +
        list(net.share_lns.parameters())
    )
    new_params = (
        list(net.superposition_module.parameters()) +
        list(net.motion_generator.parameters()) +
        list(net.value_estimator.parameters()) +
        list(net.integration_module.parameters()) +
        list(net.vision_decoder_module.parameters()) +
        list(net.feature_prediction_module.parameters())
    )

    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': LR_ENCODER},
        {'params': new_params,     'lr': LR_NEW},
    ])

    n_enc = sum(p.numel() for p in encoder_params)
    n_new = sum(p.numel() for p in new_params)
    log(f"\nパラメータ数:", log_file)
    log(f"  VisionEncoder  : {n_enc:,}  (lr={LR_ENCODER})", log_file)
    log(f"  LSTM など新規  : {n_new:,}  (lr={LR_NEW})", log_file)
    log(f"  合計           : {n_enc + n_new:,}", log_file)

    # ── 環境 / A-2 ─────────────────────────────────────────────────────────
    env_config = load_config(ENV_CONFIG_PATH)
    env = creator.create_environment(env_config.environment)
    env.init()
    env.off_display()

    green_follower = GreenFollowerAgent()

    # ── 初期エンコーダ類似度の確認 ─────────────────────────────────────────
    cos_mean, cos_std = check_encoder_similarity(net)
    log(f"\n初期エンコーダ cos_sim: mean={cos_mean:.4f}  std={cos_std:.4f}", log_file)

    log(f"\n=== train_step2_mg2 学習開始 ===\n", log_file)

    episode_losses = []

    for episode in range(EPISODES):
        env.reset()
        net.init_state(1)
        net.train()

        actor_hidden  = None
        critic_hidden = None

        v_raw, _, _, sp_0, op_0 = env.step()
        self_pos  = sp_0.copy()
        other_pos = op_0.copy()
        ep_loss   = 0.0

        for step in range(MAX_STEPS):
            v_t = preprocess_vision(v_raw)

            with torch.no_grad():
                action, _, actor_hidden = actor.sample(v_t, actor_hidden)
            a_np = action.squeeze(0).cpu().numpy()

            with torch.no_grad():
                q_raw, critic_hidden = critic(v_t, action, critic_hidden)
            q_self = q_raw / Q_SCALE

            om_np = green_follower.get_action(other_pos)

            x = {
                'self_vision': v_t,
                'self_motion': to_motion_tensor(a_np),
            }
            pred, ss, os_ = net(x, q_self, 0.0, 0.0)

            new_self_pos  = np.clip(self_pos  + a_np,  -9.5, 9.5)
            new_other_pos = np.clip(other_pos + om_np, -9.5, 9.5)
            env.self_agent.p  = new_self_pos
            env.other_agent.p = new_other_pos
            v_next_raw, _, _, _, _ = env.step()

            v_next_t = preprocess_vision(v_next_raw)
            with torch.no_grad():
                sv_enc_next = net.self_vision_encoder_module(v_next_t)
                ov_enc_next = net.other_vision_encoder_module(v_next_t)

            loss_vision     = F.l1_loss(pred['self_vision'], v_next_t)
            pred_self_feat  = net.predict_feature(ss.detach())
            pred_other_feat = net.predict_feature(os_.detach())
            loss_feat_self  = F.mse_loss(pred_self_feat,  sv_enc_next.detach())
            loss_feat_other = F.mse_loss(pred_other_feat, ov_enc_next.detach())
            loss = loss_vision + loss_feat_self + loss_feat_other

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.detach_state()
            ep_loss   += loss.item()
            v_raw      = v_next_raw
            self_pos   = new_self_pos
            other_pos  = new_other_pos

        ep_loss /= MAX_STEPS
        episode_losses.append(ep_loss)

        if (episode + 1) % LOG_INTERVAL == 0:
            recent = episode_losses[-LOG_INTERVAL:]
            mg_norm, sp_norm = get_diagnostics(net)
            cos_mean, _ = check_encoder_similarity(net, n=50)
            log(f"Episode {episode+1:5d}/{EPISODES} | "
                f"平均Loss={np.mean(recent):.5f} | "
                f"cos_sim={cos_mean:.4f} | "
                f"MG重みL2={mg_norm:.4f} | "
                f"Φ_s重みL2={sp_norm:.4f}", log_file)

    # ── 保存 ───────────────────────────────────────────────────────────────
    torch.save({
        'net_state_dict':       net.state_dict(),
        'value_estimator':      net.value_estimator.state_dict(),
        'superposition_module': net.superposition_module.state_dict(),
        'motion_generator':     net.motion_generator.state_dict(),
        'mg_input':             'ov_enc',
        'mg_hidden':            MG_HIDDEN,
        'q_scale':              Q_SCALE,
        'lr_encoder':           LR_ENCODER,
        'lr_new':               LR_NEW,
        'episodes':             EPISODES,
        'final_loss':           float(np.mean(episode_losses[-LOG_INTERVAL:])),
    }, MODEL_SAVE_PATH)

    log(f"\n学習完了。保存: {MODEL_SAVE_PATH}", log_file)
    log(f"最終平均Loss    : {np.mean(episode_losses[-LOG_INTERVAL:]):.6f}", log_file)
    mg_norm, sp_norm = get_diagnostics(net)
    cos_mean, cos_std = check_encoder_similarity(net)
    log(f"最終エンコーダ cos_sim: mean={cos_mean:.4f}  std={cos_std:.4f}", log_file)
    log(f"MG出力重みL2ノルム   : {mg_norm:.4f}", log_file)
    log(f"Φ_s LSTM重みL2ノルム : {sp_norm:.4f}", log_file)

    log_file.close()


if __name__ == '__main__':
    train()
