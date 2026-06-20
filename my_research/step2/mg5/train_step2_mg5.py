"""
train_step2_mg5.py

mg4 からの変更点:
    encoder を完全 freeze（元論文 exp3 準拠）

元論文 exp3 の Methods:
    "only Motion Generator's parameters are learned, and the parameters of
     Shared Module, Visual Encoder-1, Visual Encoder-2, and Visual Predictor
     are fixed."

mg4 では LR_ENCODER=1e-5 で encoder を微小更新していたが、
100,000 回の更新で exp1 の位置特徴が壊れていた。
mg5 では encoder を完全 freeze し、exp1 の位置特徴を保護する。

optimizer の対象:
    mg4: encoder(lr=1e-5) + 新規モジュール(lr=1e-4)
    mg5: 新規モジュールのみ(lr=1e-4)  ← encoder は optimizer から外す

保存先: results/mg5_YYYYMMDD_HHMMSS/

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 train_step2_mg5.py
"""

import sys
import os
from datetime import datetime

_HERE       = os.path.dirname(os.path.abspath(__file__))
STEP2_DIR   = os.path.abspath(os.path.join(_HERE, '..'))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../../..'))
for p in [_HERE, STEP2_DIR, MY_RESEARCH, PROJ_ROOT]:
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
from agent2_green import GreenFollowerAgent
from model_mg2 import SuperpositionNetworkWithMG2

# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES  = 1000
MAX_STEPS = 100
Q_SCALE   = 50.0
MG_HIDDEN = 64
LR_NEW    = 1e-4       # 新規モジュール用 LR
# LR_ENCODER は存在しない → encoder は optimizer に含めない

IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

LOG_INTERVAL    = 50
_TIMESTAMP      = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR         = os.path.join(STEP2_DIR, 'results', f'mg5_{_TIMESTAMP}')
MODEL_SAVE_PATH = os.path.join(RUN_DIR, 'model_mg5.pth')
LOG_PATH        = os.path.join(RUN_DIR, 'train.log')

# =========================================================================

def log(msg, f):
    print(msg); f.write(msg + '\n'); f.flush()

def preprocess_vision(v_raw):
    """(H,W,C) [0,1] → Tensor (1,C,H,W) [-1,1]  Actor/Critic は [0,1] のままで別処理"""
    v = scale_vision(v_raw.transpose(2, 0, 1))
    return torch.FloatTensor(v).unsqueeze(0).to(DEVICE)

def preprocess_vision_raw(v_raw):
    """Actor/Critic 用: scale_vision なし [0,1] のまま"""
    return torch.FloatTensor(v_raw.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

def to_motion_tensor(m):
    return torch.FloatTensor(m).unsqueeze(0).to(DEVICE)

def check_encoder_similarity(net, n=100):
    sims = []
    with torch.no_grad():
        for _ in range(n):
            v = torch.rand(1, 3, 16, 64).to(DEVICE) * 2 - 1
            f1 = net.self_vision_encoder_module(v)
            f2 = net.other_vision_encoder_module(v)
            sims.append(F.cosine_similarity(f1, f2).item())
    return float(np.mean(sims)), float(np.std(sims))

# =========================================================================

def train():
    os.makedirs(RUN_DIR, exist_ok=True)
    lf = open(LOG_PATH, 'w')

    log(f"実行ディレクトリ : {RUN_DIR}", lf)
    log(f"デバイス         : {DEVICE}", lf)
    log(f"encoder freeze   : 有効（元論文 exp3 準拠）", lf)
    log(f"scale_vision     : 有効 [-1,1]", lf)
    log(f"MG 入力          : ov_enc", lf)
    log(f"マスク戦略       : p_mask_other = (step==0 ? 0 : 1.0)", lf)
    log(f"Q_SCALE          : {Q_SCALE}", lf)
    log(f"MG_HIDDEN        : {MG_HIDDEN}", lf)
    log(f"LR_NEW           : {LR_NEW}", lf)
    log(f"EPISODES         : {EPISODES}", lf)

    # ── Actor / Critic (固定) ─────────────────────────────────────────────
    actor = ActorLSTM().to(DEVICE)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
    actor.eval()
    for p in actor.parameters(): p.requires_grad = False

    critic = CriticLSTM().to(DEVICE)
    critic.load_state_dict(torch.load(CRITIC_PATH, map_location=DEVICE))
    critic.eval()
    for p in critic.parameters(): p.requires_grad = False

    # ── SuperpositionNetworkWithMG2 ───────────────────────────────────────
    model_config = load_exp_config(IIZUKA_CONFIG_PATH)
    net = SuperpositionNetworkWithMG2(model_config, q_dim=1, mg_hidden=MG_HIDDEN).to(DEVICE)

    iizuka_ckpt = torch.load(IIZUKA_MODEL_PATH, map_location=DEVICE)
    net.load_iizuka_weights(iizuka_ckpt, DEVICE)

    # encoder を完全 freeze
    for module in [net.self_vision_encoder_module,
                   net.other_vision_encoder_module,
                   net.share_lns]:
        for p in module.parameters():
            p.requires_grad = False

    # 新規モジュールのみ optimizer に含める
    new_params = (
        list(net.superposition_module.parameters()) +
        list(net.motion_generator.parameters()) +
        list(net.value_estimator.parameters()) +
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
    log(f"  frozen (encoder) : {n_frozen:,}", lf)
    log(f"  trainable (新規) : {n_train:,}", lf)

    # ── 環境 / A-2 ──────────────────────────────────────────────────────
    env_config = load_config(ENV_CONFIG_PATH)
    env = creator.create_environment(env_config.environment)
    env.init()
    env.off_display()
    green_follower = GreenFollowerAgent()

    cos_mean, cos_std = check_encoder_similarity(net)
    log(f"\n初期エンコーダ cos_sim: mean={cos_mean:.4f}  std={cos_std:.4f}", lf)
    log(f"\n=== train_step2_mg5 学習開始 (encoder freeze) ===\n", lf)

    episode_losses = []

    for episode in range(EPISODES):
        env.reset()
        net.init_state(1)
        net.train()

        actor_h = critic_h = None
        v_raw, _, _, self_pos, other_pos = env.step()
        ep_loss = 0.0

        for step in range(MAX_STEPS):
            # Actor/Critic は [0,1] で動作（学習時と同じドメイン）
            v_raw_t = preprocess_vision_raw(v_raw)
            with torch.no_grad():
                action, _, actor_h = actor.sample(v_raw_t, actor_h)
                q_raw, critic_h    = critic(v_raw_t, action, critic_h)

            # encoder は [-1,1] で動作（exp1 学習時と同じドメイン）
            v_t    = preprocess_vision(v_raw)
            q_self = q_raw / Q_SCALE
            a_np   = action.squeeze(0).cpu().numpy()
            om_np  = green_follower.get_action(other_pos)

            p_mask_other = 0.0 if step == 0 else 1.0
            pred, ss, os_ = net(
                {'self_vision': v_t, 'self_motion': to_motion_tensor(a_np)},
                q_self, 0.0, p_mask_other
            )

            new_self  = np.clip(self_pos  + a_np,  -9.5, 9.5)
            new_other = np.clip(other_pos + om_np, -9.5, 9.5)
            env.self_agent.p  = new_self
            env.other_agent.p = new_other
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
            self_pos   = new_self
            other_pos  = new_other

        ep_loss /= MAX_STEPS
        episode_losses.append(ep_loss)

        if (episode + 1) % LOG_INTERVAL == 0:
            recent = episode_losses[-LOG_INTERVAL:]
            mg_norm = net.motion_generator.fc.weight.detach().norm().item()
            sp_norm = net.superposition_module.lstm.weight_ih.detach().norm().item()
            cos_mean, _ = check_encoder_similarity(net, n=50)
            log(f"Episode {episode+1:5d}/{EPISODES} | "
                f"Loss={np.mean(recent):.5f} | "
                f"cos_sim={cos_mean:.4f} | "
                f"MG_L2={mg_norm:.4f} | "
                f"Φs_L2={sp_norm:.4f}", lf)

    torch.save({
        'net_state_dict':       net.state_dict(),
        'value_estimator':      net.value_estimator.state_dict(),
        'superposition_module': net.superposition_module.state_dict(),
        'motion_generator':     net.motion_generator.state_dict(),
        'encoder_frozen':       True,
        'scale_vision':         True,
        'actor_critic_raw':     True,
        'mg_input':             'ov_enc',
        'p_mask_other':         'step>0 → 1.0',
        'mg_hidden':            MG_HIDDEN,
        'q_scale':              Q_SCALE,
        'lr_new':               LR_NEW,
        'episodes':             EPISODES,
        'final_loss':           float(np.mean(episode_losses[-LOG_INTERVAL:])),
    }, MODEL_SAVE_PATH)

    log(f"\n学習完了。保存: {MODEL_SAVE_PATH}", lf)
    log(f"最終平均Loss : {np.mean(episode_losses[-LOG_INTERVAL:]):.6f}", lf)
    cos_mean, cos_std = check_encoder_similarity(net)
    log(f"最終 cos_sim : mean={cos_mean:.4f}  std={cos_std:.4f}", lf)
    lf.close()

if __name__ == '__main__':
    train()
