"""
train_step2_v2.py

Step 2 学習スクリプト
─────────────────────────────────────────────────
A-1 : SAC+LSTM (Step1 で学習済み・固定)
A-2 : GreenFollower (緑ランドマーク (-9,-9) へ直接向かう)

新設計:
    飯塚ネットワーク Φ_s にQ値を追加入力して再学習
        Process-1 (A-1): Q値 = A-1 の Critic が出力する値 (固定)
        Process-2 (A-2): Q値 = ValueEstimator が推測する値 (学習)
    損失 = 視覚予測誤差 + 特徴予測誤差 (元論文と同じ)

実行コマンド:
    MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run python3 train_step2_v2.py

─────────────────────────────────────────────────
凍結ルール:
    凍結 (飯塚重みをそのまま使う):
        - VisionEncoder (self / other) ← 視覚特徴は再利用
        - share_lns

    学習 (新しい h¹/h² に適応させる):
        - SuperpositionModuleWithQ (Φ_s) ← 67次元入力・ランダム初期化
        - ValueEstimator              ← 新規モジュール
        - IntegrationModule           ← 新Φ_sのh¹/h²を受けるため再適応
        - VisionDecoderModule         ← 同上
        - FeaturePredictionModule     ← 同上

    ※ 元の 00200.pth は一切変更しない。model_q.pth として別保存。
    ※ RETRAIN_IIZUKA=True に変えると VisionEncoder も学習対象になる。

─────────────────────────────────────────────────
Q値の正規化について:
    A-1 Critic の出力は ~38 程度だが、VisionEncoder 出力は
    LayerNorm 後で ±1 程度。そのままでは Φ_s の LSTM が Q 入力を
    無視する方向に収束するため、Q_SCALE で割って正規化する。
─────────────────────────────────────────────────
"""

import sys
import os

# --- sys.path 設定 --------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(_HERE, '../../'))   # superposition/
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '../'))    # my_research/
# PROJ_ROOT を先頭に置く（my_research/model.py より model/ パッケージを優先）
for p in [_HERE, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)
# -------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from simulation import creator
from simulation.util import load_config
from config_util import load_config as load_exp_config

from rl_agent_sac import ActorLSTM, CriticLSTM
from agent2_green import GreenFollowerAgent
from model_q import SuperpositionNetworkWithQ  # step2/ 内のモジュール

# =========================================================================
# ハイパーパラメータ
# =========================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES  = 1000  # 学習エピソード数
MAX_STEPS = 100   # 1エピソードの最大ステップ数
LR        = 1e-4  # 学習率

# --- Q値の正規化スケール ---
# A-1 Critic の出力 ~38 を VisionEncoder 出力のスケール (~±1) に合わせる
Q_SCALE = 50.0

# --- VisionEncoder も再学習するか (教授との合意後に True に変更) ---
RETRAIN_IIZUKA = False

# --- ファイルパス ---
IIZUKA_MODEL_PATH  = os.path.join(PROJ_ROOT, 'data/result/exp1_l1/0/model/00200.pth')
IIZUKA_CONFIG_PATH = os.path.join(PROJ_ROOT, 'config/model/SuperpositionNetworkFeaturePrediction/default.yml')
ACTOR_PATH         = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
CRITIC_PATH        = os.path.join(MY_RESEARCH, 'rl_model_v6_critic.pth')
ENV_CONFIG_PATH    = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')

SAVE_DIR        = _HERE
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'model_q.pth')
LOG_INTERVAL    = 50   # ログ出力間隔 (エピソード数)

# =========================================================================
# ユーティリティ
# =========================================================================

def preprocess_vision(v_raw: np.ndarray) -> torch.Tensor:
    """HWC numpy → BCHW tensor"""
    return torch.FloatTensor(
        v_raw.transpose(2, 0, 1)
    ).unsqueeze(0).to(DEVICE)


def to_motion_tensor(m_np: np.ndarray) -> torch.Tensor:
    """(2,) numpy → (1,2) tensor"""
    return torch.FloatTensor(m_np).unsqueeze(0).to(DEVICE)


def get_q_weight_norm(net: SuperpositionNetworkWithQ) -> float:
    """
    Φ_s の LSTM の weight_ih のうち、Q入力次元（最後の1列）の L2 ノルム。
    この値が小さい場合、Φ_s が Q 入力を実質的に無視している可能性が高い。
    """
    w = net.superposition_module.lstm.weight_ih  # (4*hidden, input_size)
    q_col = w[:, -1]                             # Q 次元に対応する列
    return q_col.detach().norm().item()


# =========================================================================
# 学習ループ
# =========================================================================

def train():
    print(f"デバイス        : {DEVICE}")
    print(f"RETRAIN_IIZUKA  : {RETRAIN_IIZUKA}  (True=VisionEncoderも学習)")
    print(f"Q_SCALE         : {Q_SCALE}")
    print(f"EPISODES        : {EPISODES}")

    # ---------------------------------------------------------------
    # 1. A-1 の Actor / Critic 読み込み (固定)
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # 2. SuperpositionNetworkWithQ の初期化
    #    ※ 元の 00200.pth は読み取り専用。model_q.pth に別途保存する。
    # ---------------------------------------------------------------
    model_config = load_exp_config(IIZUKA_CONFIG_PATH)
    net = SuperpositionNetworkWithQ(model_config).to(DEVICE)

    iizuka_ckpt = torch.load(IIZUKA_MODEL_PATH, map_location=DEVICE)
    net.load_iizuka_weights(iizuka_ckpt, DEVICE)

    # ---------------------------------------------------------------
    # 3. 凍結設定
    #    VisionEncoder は飯塚の重みをそのまま使う（転送済み）
    #    Integration/Decoder/FeaturePrediction は新Φ_sに合わせて再適応
    # ---------------------------------------------------------------
    if not RETRAIN_IIZUKA:
        for mod in [net.self_vision_encoder_module,
                    net.other_vision_encoder_module,
                    net.share_lns]:
            for p in mod.parameters():
                p.requires_grad = False

    trainable = [p for p in net.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_frozen    = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    print(f"学習パラメータ数: {n_trainable:,}  凍結: {n_frozen:,}")
    optimizer = optim.Adam(trainable, lr=LR)

    # ---------------------------------------------------------------
    # 4. 環境 / A-2 の初期化
    # ---------------------------------------------------------------
    env_config = load_config(ENV_CONFIG_PATH)
    env = creator.create_environment(env_config.environment)
    env.init()
    env.off_display()

    green_follower = GreenFollowerAgent()

    # ---------------------------------------------------------------
    # 5. 学習ループ
    # ---------------------------------------------------------------
    print(f"\n=== Step2 学習開始 (episodes={EPISODES}, max_steps={MAX_STEPS}) ===\n")

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
            v_t = preprocess_vision(v_raw)  # (1, 3, 16, 64)

            # --- A-1: 行動サンプリング ---
            with torch.no_grad():
                action, _, actor_hidden = actor.sample(v_t, actor_hidden)
            a_np = action.squeeze(0).cpu().numpy()

            # --- A-1: Q値取得 → 正規化 ---
            # Critic出力 (~38) を VisionEncoder スケール (~±1) に合わせる
            with torch.no_grad():
                q_raw, critic_hidden = critic(v_t, action, critic_hidden)
            q_self = q_raw / Q_SCALE  # (1, 1)

            # --- A-2: GreenFollower 運動 ---
            om_np = green_follower.get_action(other_pos)
            sm_t  = to_motion_tensor(a_np)   # (1, 2)
            om_t  = to_motion_tensor(om_np)  # (1, 2)

            # --- 順伝播 ---
            x = {
                'self_vision':  v_t,
                'self_motion':  sm_t,
                'other_motion': om_t,
            }
            pred, ss, os = net(x, q_self,
                               p_mask_vision_self=0.0,
                               p_mask_vision_other=0.0)

            # --- 次の視覚を取得 ---
            new_self_pos  = np.clip(self_pos  + a_np,  -9.5, 9.5)
            new_other_pos = np.clip(other_pos + om_np, -9.5, 9.5)
            env.self_agent.p  = new_self_pos
            env.other_agent.p = new_other_pos
            v_next_raw, _, _, _, _ = env.step()

            v_next_t = preprocess_vision(v_next_raw)
            with torch.no_grad():
                sv_enc_next = net.self_vision_encoder_module(v_next_t)
                ov_enc_next = net.other_vision_encoder_module(v_next_t)

            # --- 損失計算 ---
            loss_vision     = F.l1_loss(pred['self_vision'], v_next_t)
            pred_self_feat  = net.predict_feature(ss.detach())
            pred_other_feat = net.predict_feature(os.detach())
            loss_feat_self  = F.mse_loss(pred_self_feat,  sv_enc_next.detach())
            loss_feat_other = F.mse_loss(pred_other_feat, ov_enc_next.detach())
            loss = loss_vision + loss_feat_self + loss_feat_other

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            net.superposition_module.detach_state()
            ep_loss  += loss.item()
            v_raw     = v_next_raw
            self_pos  = new_self_pos
            other_pos = new_other_pos

        ep_loss /= MAX_STEPS
        episode_losses.append(ep_loss)

        if (episode + 1) % LOG_INTERVAL == 0:
            recent = episode_losses[-LOG_INTERVAL:]
            q_norm = get_q_weight_norm(net)
            print(f"Episode {episode+1:5d}/{EPISODES} | "
                  f"平均Loss={np.mean(recent):.5f} | "
                  f"Q入力重みL2={q_norm:.4f}")

    # ---------------------------------------------------------------
    # 6. モデル保存 (元の 00200.pth は変更しない)
    # ---------------------------------------------------------------
    torch.save({
        'net_state_dict':       net.state_dict(),
        'value_estimator':      net.value_estimator.state_dict(),
        'superposition_module': net.superposition_module.state_dict(),
        'retrain_iizuka':       RETRAIN_IIZUKA,
        'q_scale':              Q_SCALE,
        'episodes':             EPISODES,
        'final_loss':           float(np.mean(episode_losses[-LOG_INTERVAL:])),
    }, MODEL_SAVE_PATH)

    print(f"\n学習完了。モデルを保存: {MODEL_SAVE_PATH}")
    print(f"最終平均Loss    : {np.mean(episode_losses[-LOG_INTERVAL:]):.6f}")
    print(f"Q入力重みL2ノルム: {get_q_weight_norm(net):.4f}")


if __name__ == '__main__':
    train()
