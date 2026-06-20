"""
train_step3_phase2.py

Phase 2: Motion Generator (MG) 専用訓練。

【方針】
    飯塚モデル (exp1_l1) の SM・encoder 等を全て frozen にして
    MotionGeneratorModule だけを訓練する。
    A-2 は MultiPrefAgent（4 選好でランダム移動）。

【アーキテクチャ】
    元論文 exp3 の SuperpositionNetworkMotionGenerationFeaturePrediction を使用。
    これは SuperpositionNetworkFeaturePrediction (exp1) に MG を追加したもの。

    forward() での処理順:
        1. ov_enc = other_encoder(v_t)           ← マスク前（MG が本物の視覚を受け取る）
        2. om = MG(ov_enc)                       ← MG が A-2 の運動を推定
        3. sv_enc = mask(sv_enc, p_mask)         ← step>0 でマスク (p=1.0)
        4. ov_enc = mask(ov_enc, p_mask)         ← step>0 でマスク (p=1.0)
        5. SM(sv_enc, sm, ov_enc=0, om) → h¹,h² ← frozen SM が om を使って追跡

【損失】
    L_vision + L_feat_self + L_feat_other（exp1 と同じ）
    SM が frozen でも om_generated → SM の計算グラフを通じて
    gradient が MG まで届く（frozen パラメータは更新されないが勾配は流れる）。

【学習方式】
    BPTT（エピソード全体で loss 積算、1回 backward）。

保存先: step3/results/phase2_YYYYMMDD_HHMMSS/

実行コマンド（コンテナ内）:
    cd /work
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/phase2/train_step3_phase2.py \\
    2>&1 | tee my_research/step3/results/train_phase2.log
"""

import sys, os
from datetime import datetime
from collections import Counter

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

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config
from util import scale_vision
from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_multi_pref import MultiPrefAgent
from model.model import SuperpositionNetworkMotionGenerationFeaturePrediction

# =========================================================================
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPISODES  = 5000
MAX_STEPS = 100
Q_SCALE   = 50.0
LR_MG     = 1e-3   # 元論文 exp3 に合わせて 0.001

# 前回チェックポイントから続行する場合はパスを指定、None なら飯塚 checkpoint から開始
RESUME_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results/phase2_20260615_193019/model_phase2.pth'
)

IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkMotionGenerationFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

LOG_INTERVAL    = 100
_TIMESTAMP      = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR         = os.path.join(STEP3_DIR, 'results', f'phase2_{_TIMESTAMP}')
MODEL_SAVE_PATH = os.path.join(RUN_DIR, 'model_phase2.pth')
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
    log(f"Phase            : 2 (MG 専用訓練、SM frozen)", lf)
    log(f"基盤モデル       : {IIZUKA_MODEL_PATH}", lf)
    log(f"A-2 行動         : MultiPrefAgent (4 選好、エピソードごとランダム)", lf)
    log(f"マスク戦略       : step>0 → p_mask=1.0 (self & other)", lf)
    log(f"学習方式         : BPTT (エピソード全体で loss 積算、1回 backward)", lf)
    log(f"LR_MG            : {LR_MG}  (AdamW, weight_decay=0.01)", lf)
    log(f"再開 checkpoint  : {RESUME_CHECKPOINT if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT) else 'なし (飯塚から開始)'}", lf)
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

    # ── モデル構築: SuperpositionNetworkMotionGenerationFeaturePrediction ─
    model_config = load_exp_config(IIZUKA_CONFIG_PATH)
    net = SuperpositionNetworkMotionGenerationFeaturePrediction(model_config).to(DEVICE)

    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        # 前回チェックポイントから全パラメータ（MG を含む）を読み込む
        resume_ckpt = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        resume_state = resume_ckpt.get('net_state_dict', resume_ckpt)
        net.load_state_dict(resume_state)
        log(f"\nチェックポイントから再開: {RESUME_CHECKPOINT}", lf)
    else:
        # 飯塚 checkpoint から SM・encoder 等を読み込む（MG はランダム初期化）
        iizuka_ckpt = torch.load(IIZUKA_MODEL_PATH, map_location=DEVICE)
        iizuka_state = iizuka_ckpt.get('model', iizuka_ckpt)
        missing, unexpected = net.load_state_dict(iizuka_state, strict=False)
        log(f"\n飯塚 checkpoint 読み込み:", lf)
        log(f"  missing (MG 等): {missing}", lf)
        log(f"  unexpected     : {unexpected}", lf)

    # MG 以外を全て frozen
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
    optimizer  = optim.AdamW(mg_params, lr=LR_MG, weight_decay=0.01)

    n_frozen  = sum(p.numel() for m in freeze_modules for p in m.parameters())
    n_mg      = sum(p.numel() for p in mg_params)
    log(f"\nパラメータ数:", lf)
    log(f"  frozen (SM 等) : {n_frozen:,}", lf)
    log(f"  trainable (MG) : {n_mg:,}", lf)

    # ── 環境 ─────────────────────────────────────────────────────────────
    env_config = load_config(ENV_CONFIG_PATH)
    env = creator.create_environment(env_config.environment)
    env.init()
    env.off_display()
    multi_agent = MultiPrefAgent()

    log(f"\n=== Phase 2 学習開始 ===\n", lf)

    episode_losses = []
    pref_counter   = Counter()
    # MG 相関追跡用バッファ (LOG_INTERVAL エピソード分)
    buf_om_pred = []
    buf_om_true = []

    for episode in range(EPISODES):
        pref_name = multi_agent.reset()
        pref_counter[pref_name] += 1

        env.reset()
        net.init_state(1)
        net.train()

        actor_h = critic_h = None
        v_raw, _, _, self_pos, other_pos = env.step()
        ep_total_loss = torch.tensor(0.0, device=DEVICE)

        for step in range(MAX_STEPS):
            v_raw_t = preprocess_vision_raw(v_raw)
            with torch.no_grad():
                action, _, actor_h = actor.sample(v_raw_t, actor_h)
                q_raw, critic_h    = critic(v_raw_t, action, critic_h)

            v_t   = preprocess_vision(v_raw)
            a_np  = action.squeeze(0).cpu().numpy()
            om_np = multi_agent.get_action(other_pos)

            # step>0 でマスク（MG が om を生成してから ov_enc をマスク）
            p_mask = 0.0 if step == 0 else 1.0
            pred = net(
                {'self_vision': v_t, 'self_motion': to_tensor(a_np)},
                p_mask_vision_self=p_mask,
                p_mask_vision_other=p_mask,
            )

            # h¹/h² は state から取得
            state = net.superposition_module.state
            ss = state['self'].hidden    # (1, 128)
            os_ = state['other'].hidden  # (1, 128)

            # MG 相関追跡
            om_pred_np = pred['other_motion'].detach().squeeze(0).cpu().numpy()
            buf_om_pred.append(om_pred_np)
            buf_om_true.append(om_np)

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
            # os_ は detach しない → loss_feat_other → h² → SM(frozen) → om → MG の勾配パス
            pred_self_feat  = net.predict_feature(ss.detach())
            pred_other_feat = net.predict_feature(os_)
            loss_feat_self  = F.mse_loss(pred_self_feat,  sv_enc_next.detach())
            loss_feat_other = F.mse_loss(pred_other_feat, ov_enc_next.detach())
            step_loss = loss_vision + loss_feat_self + loss_feat_other

            ep_total_loss = ep_total_loss + step_loss

            v_raw     = v_next_raw
            self_pos  = new_self
            other_pos = new_other

        optimizer.zero_grad()
        (ep_total_loss / MAX_STEPS).backward()
        torch.nn.utils.clip_grad_norm_(mg_params, max_norm=1.0)
        optimizer.step()

        net.detach_state()

        ep_loss = (ep_total_loss / MAX_STEPS).item()
        episode_losses.append(ep_loss)

        if (episode + 1) % LOG_INTERVAL == 0:
            recent   = episode_losses[-LOG_INTERVAL:]
            mg_norm  = net.motion_generator_module.fc.weight.detach().norm().item()
            dist_str = ' '.join(f"{k}:{v}" for k, v in sorted(pref_counter.items()))
            # Pearson r (バッファ内)
            arr_pred = np.array(buf_om_pred)
            arr_true = np.array(buf_om_true)
            from scipy.stats import pearsonr as _pr
            rx = _pr(arr_pred[:, 0], arr_true[:, 0])[0]
            ry = _pr(arr_pred[:, 1], arr_true[:, 1])[0]
            buf_om_pred.clear(); buf_om_true.clear()
            log(f"Episode {episode+1:5d}/{EPISODES} | "
                f"Loss={np.mean(recent):.5f} | "
                f"MG_L2={mg_norm:.4f} | "
                f"r=({rx:.3f},{ry:.3f}) | "
                f"pref=[{dist_str}]", lf)

    torch.save({
        'net_state_dict':           net.state_dict(),
        'motion_generator_module':  net.motion_generator_module.state_dict(),
        'iizuka_base':              IIZUKA_MODEL_PATH,
        'frozen':                   'all except motion_generator_module',
        'a2_agent':                 'MultiPrefAgent',
        'p_mask':                   'step>0 → 1.0 (self & other)',
        'training':                 'BPTT episode-level',
        'lr_mg':                    LR_MG,
        'episodes':                 EPISODES,
        'max_steps':                MAX_STEPS,
        'final_loss':               float(np.mean(episode_losses[-LOG_INTERVAL:])),
        'pref_dist':                dict(pref_counter),
        'phase':                    2,
    }, MODEL_SAVE_PATH)

    log(f"\n学習完了。保存: {MODEL_SAVE_PATH}", lf)
    log(f"最終平均 Loss : {np.mean(episode_losses[-LOG_INTERVAL:]):.6f}", lf)
    log(f"選好分布      : {dict(pref_counter)}", lf)
    lf.close()


if __name__ == '__main__':
    train()
