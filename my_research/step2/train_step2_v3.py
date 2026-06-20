"""
train_step2_v3.py

Step 2 学習スクリプト（全体ファインチューニング版）
─────────────────────────────────────────────────
v2 との違い:
    VisionEncoder も含めて全パラメータを学習対象にする。
    ただし飯塚モデルの重みを出発点として使うため、
    ゼロからではなく「ファインチューニング」になる。

    VisionEncoder は飯塚の学習済み重みから微小に更新されるだけなので
    視覚理解は保たれつつ、Step2 のタスクに最適化される。

学習率の設定（差分学習率）:
    VisionEncoder, share_lns   : LR_ENCODER = 1e-5  （小さく、壊さない）
    LSTM, ValueEstimator,
    Integration, Decoder,
    FeaturePrediction          : LR_NEW     = 1e-4  （通常）

初期化:
    VisionEncoder, IntegrationModule, VisionDecoder,
    FeaturePredictionModule    : 飯塚モデル (00200.pth) の重みで初期化
    SuperpositionModuleWithQ   : ランダム初期化（Q入力追加で互換性なし）
    ValueEstimator             : ランダム初期化（新規モジュール）

保存先: model_q_v3.pth （v2 の model_q.pth は変更しない）

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 train_step2_v3.py
─────────────────────────────────────────────────
"""

import sys
import os

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
from model_q import SuperpositionNetworkWithQ

# =========================================================================
# ハイパーパラメータ
# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES  = 1000
MAX_STEPS = 100
Q_SCALE   = 50.0

# 差分学習率
LR_ENCODER = 1e-5   # VisionEncoder（飯塚の知識を保ちながら微調整）
LR_NEW     = 1e-4   # LSTM・ValueEstimator・Integration・Decoder・FP

# ファイルパス
IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

MODEL_SAVE_PATH = os.path.join(_HERE, 'model_q_v3.pth')   # v2 の pth は触らない
LOG_INTERVAL    = 50

# =========================================================================
# ユーティリティ
# =========================================================================

def preprocess_vision(v_raw):
    return torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

def to_motion_tensor(m_np):
    return torch.FloatTensor(m_np).unsqueeze(0).to(DEVICE)

def get_q_weight_norm(net):
    w = net.superposition_module.lstm.weight_ih
    return w[:, -1].detach().norm().item()

# =========================================================================
# 学習ループ
# =========================================================================

def train():
    print(f"デバイス         : {DEVICE}")
    print(f"Q_SCALE          : {Q_SCALE}")
    print(f"LR_ENCODER       : {LR_ENCODER}  (VisionEncoder)")
    print(f"LR_NEW           : {LR_NEW}  (LSTM / ValueEstimator など)")
    print(f"EPISODES         : {EPISODES}")
    print(f"保存先           : {MODEL_SAVE_PATH}")

    # ── A-1 の Actor / Critic（固定） ──────────────────────────────────────
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

    # ── SuperpositionNetworkWithQ の初期化 ─────────────────────────────────
    model_config = load_exp_config(IIZUKA_CONFIG_PATH)
    net = SuperpositionNetworkWithQ(model_config).to(DEVICE)

    # 飯塚の重みで初期化（VisionEncoder / Integration / Decoder / FP）
    # SuperpositionModuleWithQ と ValueEstimator はランダム初期化のまま
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
    print(f"\nパラメータ数:")
    print(f"  VisionEncoder  : {n_enc:,}  (lr={LR_ENCODER})")
    print(f"  LSTM など新規  : {n_new:,}  (lr={LR_NEW})")
    print(f"  合計           : {n_enc + n_new:,}")

    # ── 環境 / A-2 ─────────────────────────────────────────────────────────
    env_config = load_config(ENV_CONFIG_PATH)
    env = creator.create_environment(env_config.environment)
    env.init()
    env.off_display()

    green_follower = GreenFollowerAgent()

    # ── 学習ループ ─────────────────────────────────────────────────────────
    print(f"\n=== Step2 v3 学習開始 ===\n")

    episode_losses = []

    for episode in range(EPISODES):
        env.reset()
        net.superposition_module.init_state(1)
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
            sm_t  = to_motion_tensor(a_np)
            om_t  = to_motion_tensor(om_np)

            x = {
                'self_vision':  v_t,
                'self_motion':  sm_t,
                'other_motion': om_t,
            }
            pred, ss, os_ = net(x, q_self,
                                p_mask_vision_self=0.0,
                                p_mask_vision_other=0.0)

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

            net.superposition_module.detach_state()
            ep_loss   += loss.item()
            v_raw      = v_next_raw
            self_pos   = new_self_pos
            other_pos  = new_other_pos

        ep_loss /= MAX_STEPS
        episode_losses.append(ep_loss)

        if (episode + 1) % LOG_INTERVAL == 0:
            recent = episode_losses[-LOG_INTERVAL:]
            q_norm = get_q_weight_norm(net)
            print(f"Episode {episode+1:5d}/{EPISODES} | "
                  f"平均Loss={np.mean(recent):.5f} | "
                  f"Q入力重みL2={q_norm:.4f}")

    # ── 保存 ───────────────────────────────────────────────────────────────
    torch.save({
        'net_state_dict':       net.state_dict(),
        'value_estimator':      net.value_estimator.state_dict(),
        'superposition_module': net.superposition_module.state_dict(),
        'retrain_iizuka':       True,
        'q_scale':              Q_SCALE,
        'lr_encoder':           LR_ENCODER,
        'lr_new':               LR_NEW,
        'episodes':             EPISODES,
        'final_loss':           float(np.mean(episode_losses[-LOG_INTERVAL:])),
    }, MODEL_SAVE_PATH)

    print(f"\n学習完了。保存: {MODEL_SAVE_PATH}")
    print(f"最終平均Loss    : {np.mean(episode_losses[-LOG_INTERVAL:]):.6f}")
    print(f"Q入力重みL2ノルム: {get_q_weight_norm(net):.4f}")


if __name__ == '__main__':
    train()
