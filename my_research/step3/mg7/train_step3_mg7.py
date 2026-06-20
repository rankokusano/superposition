"""
train_step3_mg7.py

mg6 からの変更点:
    GreenFollowerAgent → MultiPrefAgent
    各エピソードの開始時に A-2 の選好をランダムに選択（Red/Green/Blue/Cyan）

【変更しないもの】
    - アーキテクチャ (SuperpositionNetworkWithMG3)
    - encoder freeze（元論文 exp3 準拠）
    - 損失関数: L_vision + L_feat_self + L_feat_other
    - マスク戦略: step==0 のみ p_mask_other=0、以降は 1.0
    - Actor/Critic の使い方（A-1 の行動と Q 値取得）
    - VE 入力: (ov_enc, om_generated)

【変更点の意図】
    mg6 では A-2 が常に Green へ向かうため：
        (1) A-1 (Red 付近) と A-2 (Green 付近) の位置が構造的に反相関する
        (2) h² → A-1 位置の R² が 0.814 と高く、自他分離の指標が正直でない
        (3) VE が「A-2 は常に Green 方向」を暗記するだけで選好推論をしない
    mg7 では A-2 の選好をエピソードごとにランダム化することで：
        (1) A-1 と A-2 の位置相関が崩れ、R² 指標が正直になる
        (2) VE が「om_generated の方向 → Q 値」を汎化的に学習するよう強制される
        (3) 評価時に各選好で Q-map を描くことで、VE が選好推論できているかを検証できる

保存先: step3/results/mg7_YYYYMMDD_HHMMSS/

実行コマンド（コンテナ内）:
    cd /work
    Xvfb :99 -screen 0 1024x768x24 & sleep 1
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/mg7/train_step3_mg7.py \\
    2>&1 | tee my_research/step3/results/train_mg7.log
"""

import sys
import os
from datetime import datetime
from collections import Counter

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
from agent2_multi_pref import MultiPrefAgent
from model_mg3 import SuperpositionNetworkWithMG3

# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES  = 1000
MAX_STEPS = 100
Q_SCALE   = 50.0
MG_HIDDEN = 64
LR_NEW    = 1e-4

IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

LOG_INTERVAL    = 50
_TIMESTAMP      = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR         = os.path.join(STEP3_DIR, 'results', f'mg7_{_TIMESTAMP}')
MODEL_SAVE_PATH = os.path.join(RUN_DIR, 'model_mg7.pth')
LOG_PATH        = os.path.join(RUN_DIR, 'train.log')

# =========================================================================

def log(msg, f):
    print(msg); f.write(msg + '\n'); f.flush()

def preprocess_vision(v_raw):
    v = scale_vision(v_raw.transpose(2, 0, 1))
    return torch.FloatTensor(v).unsqueeze(0).to(DEVICE)

def preprocess_vision_raw(v_raw):
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
    log(f"A-2 行動         : MultiPrefAgent（エピソードごとにランドマークをランダム選択）", lf)
    log(f"VE 入力          : (ov_enc, om_generated)", lf)
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

    # ── SuperpositionNetworkWithMG3 ───────────────────────────────────────
    model_config = load_exp_config(IIZUKA_CONFIG_PATH)
    net = SuperpositionNetworkWithMG3(
        model_config, q_dim=1, mg_hidden=MG_HIDDEN).to(DEVICE)

    iizuka_ckpt = torch.load(IIZUKA_MODEL_PATH, map_location=DEVICE)
    net.load_iizuka_weights(iizuka_ckpt, DEVICE)

    # encoder を完全 freeze
    for module in [net.self_vision_encoder_module,
                   net.other_vision_encoder_module,
                   net.share_lns]:
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
    multi_agent = MultiPrefAgent()

    cos_mean, cos_std = check_encoder_similarity(net)
    log(f"\n初期エンコーダ cos_sim: mean={cos_mean:.4f}  std={cos_std:.4f}", lf)
    log(f"\n=== train_step3_mg7 学習開始 (encoder freeze, MultiPrefAgent) ===\n", lf)

    episode_losses = []
    pref_counter = Counter()

    for episode in range(EPISODES):
        pref_name = multi_agent.reset()
        pref_counter[pref_name] += 1

        env.reset()
        net.init_state(1)
        net.train()

        actor_h = critic_h = None
        v_raw, _, _, self_pos, other_pos = env.step()
        ep_loss = 0.0

        for step in range(MAX_STEPS):
            v_raw_t = preprocess_vision_raw(v_raw)
            with torch.no_grad():
                action, _, actor_h = actor.sample(v_raw_t, actor_h)
                q_raw, critic_h    = critic(v_raw_t, action, critic_h)

            v_t    = preprocess_vision(v_raw)
            q_self = q_raw / Q_SCALE
            a_np   = action.squeeze(0).cpu().numpy()
            om_np  = multi_agent.get_action(other_pos)

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
            ep_loss  += loss.item()
            v_raw     = v_next_raw
            self_pos  = new_self
            other_pos = new_other

        ep_loss /= MAX_STEPS
        episode_losses.append(ep_loss)

        if (episode + 1) % LOG_INTERVAL == 0:
            recent  = episode_losses[-LOG_INTERVAL:]
            mg_norm = net.motion_generator.fc.weight.detach().norm().item()
            sp_norm = net.superposition_module.lstm.weight_ih.detach().norm().item()
            cos_mean, _ = check_encoder_similarity(net, n=50)
            dist_str = ' '.join(f"{k}:{v}" for k, v in sorted(pref_counter.items()))
            log(f"Episode {episode+1:5d}/{EPISODES} | "
                f"Loss={np.mean(recent):.5f} | "
                f"cos_sim={cos_mean:.4f} | "
                f"MG_L2={mg_norm:.4f} | "
                f"Φs_L2={sp_norm:.4f} | "
                f"pref_dist=[{dist_str}]", lf)

    torch.save({
        'net_state_dict':       net.state_dict(),
        'value_estimator':      net.value_estimator.state_dict(),
        'superposition_module': net.superposition_module.state_dict(),
        'motion_generator':     net.motion_generator.state_dict(),
        'encoder_frozen':       True,
        'scale_vision':         True,
        'actor_critic_raw':     True,
        've_input':             'ov_enc+om_generated',
        'mg_input':             'ov_enc',
        'p_mask_other':         'step>0 → 1.0',
        'mg_hidden':            MG_HIDDEN,
        'q_scale':              Q_SCALE,
        'lr_new':               LR_NEW,
        'episodes':             EPISODES,
        'a2_agent':             'MultiPrefAgent',
        'final_loss':           float(np.mean(episode_losses[-LOG_INTERVAL:])),
        'pref_dist':            dict(pref_counter),
    }, MODEL_SAVE_PATH)

    log(f"\n学習完了。保存: {MODEL_SAVE_PATH}", lf)
    log(f"最終平均Loss : {np.mean(episode_losses[-LOG_INTERVAL:]):.6f}", lf)
    cos_mean, cos_std = check_encoder_similarity(net)
    log(f"最終 cos_sim : mean={cos_mean:.4f}  std={cos_std:.4f}", lf)
    log(f"選好分布     : {dict(pref_counter)}", lf)
    lf.close()


if __name__ == '__main__':
    train()
