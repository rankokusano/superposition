"""
collect_data_phase2.py

Phase 2 用オフラインデータセットを収集する。

【設定】
    A-1 : 訓練済み RL actor (SAC)
    A-2 : MultiPrefAgent（4 選好、エピソードごとにランダム）

【出力形式】
    既存の data/data/*/data.h5 と同じ構造。
    train/test 各スプリットに以下を保存:
        self_vision   : (N, seq_length, H, W, C)
        self_motion   : (N, seq_length, 2)
        self_position : (N, seq_length, 2)
        other_motion  : (N, seq_length, 2)
        other_position: (N, seq_length, 2)
        pref_label    : (N,)  ← 我々の研究用追加データ（整数 0-3）
        pref_name     : (N,)  ← 選好名文字列

実行コマンド（コンテナ内）:
    cd /work
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/collect_data_phase2.py \\
    2>&1 | tee my_research/step3/results/collect_phase2.log
"""

import sys, os, argparse

_HERE       = os.path.dirname(os.path.abspath(__file__))
MY_RESEARCH = os.path.abspath(os.path.join(_HERE, '..'))
PROJ_ROOT   = os.path.abspath(os.path.join(_HERE, '../..'))
for p in [_HERE, MY_RESEARCH, PROJ_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
import h5py

from simulation import creator
from simulation.util import load_config
from rl_agent_sac import ActorLSTM
from agent2_multi_pref import MultiPrefAgent, LANDMARK_NAMES

# =========================================================================
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORLD_HALF  = 9.5
SEQ_LENGTH  = 101     # 元論文と同じ（100 ステップ + 最終観測）
N_TRAIN     = 2100    # 元論文 exp3 と同じ
N_TEST      = 220     # 元論文 exp3 と同じ

ACTOR_PATH      = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
ENV_CONFIG_PATH = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')
SAVE_DIR        = os.path.join(PROJ_ROOT, 'data/data/self_random_other_stay_multi_pref')
SAVE_PATH       = os.path.join(SAVE_DIR, 'data.h5')

# =========================================================================

def collect_split(env, actor, multi_agent, h5_file, mode, n_episodes, print_interval=100):
    H = env.world.config.camera.agent.height
    W = env.world.config.camera.agent.width
    C = 3

    grp = h5_file.require_group(mode)
    ds_sv = grp.create_dataset('self_vision',    (n_episodes, SEQ_LENGTH, H, W, C), dtype='f')
    ds_sm = grp.create_dataset('self_motion',    (n_episodes, SEQ_LENGTH, 2),        dtype='f')
    ds_sp = grp.create_dataset('self_position',  (n_episodes, SEQ_LENGTH, 2),        dtype='f')
    ds_om = grp.create_dataset('other_motion',   (n_episodes, SEQ_LENGTH, 2),        dtype='f')
    ds_op = grp.create_dataset('other_position', (n_episodes, SEQ_LENGTH, 2),        dtype='f')
    ds_pl = grp.create_dataset('pref_label',     (n_episodes,),                      dtype='i')
    dt    = h5py.special_dtype(vlen=str)
    ds_pn = grp.create_dataset('pref_name',      (n_episodes,),                      dtype=dt)

    pref_counts = {name: 0 for name in LANDMARK_NAMES}

    for n in range(n_episodes):
        pref_name = multi_agent.reset()
        pref_idx  = LANDMARK_NAMES.index(pref_name)
        pref_counts[pref_name] += 1

        env.reset()
        actor_h = None

        sp = env.self_agent.p.copy()
        op = env.other_agent.p.copy()

        for t in range(SEQ_LENGTH):
            # 現在位置での視覚観測
            v = env.capture()   # (H, W, C)

            # A-1: RL actor で行動決定
            v_tensor = torch.FloatTensor(
                v.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                sm_tensor, _, actor_h = actor.sample(v_tensor, actor_h)
            sm = sm_tensor.squeeze(0).cpu().numpy()   # (2,)

            # A-2: MultiPrefAgent で行動決定
            om = multi_agent.get_action(op)           # (2,)

            # 保存
            ds_sv[n, t] = v
            ds_sm[n, t] = sm
            ds_sp[n, t] = sp
            ds_om[n, t] = om
            ds_op[n, t] = op

            # 位置更新
            sp = np.clip(sp + sm, -WORLD_HALF, WORLD_HALF)
            op = np.clip(op + om, -WORLD_HALF, WORLD_HALF)
            env.self_agent.p  = sp.copy()
            env.other_agent.p = op.copy()

        ds_pl[n] = pref_idx
        ds_pn[n] = pref_name

        if (n + 1) % print_interval == 0:
            dist = ' '.join(f"{k}:{v}" for k, v in sorted(pref_counts.items()))
            print(f"  [{mode}] {n+1:5d}/{n_episodes}  pref=[{dist}]")

    print(f"  [{mode}] 完了: {n_episodes} episodes  pref={pref_counts}")
    return pref_counts


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if os.path.exists(SAVE_PATH):
        print(f"警告: {SAVE_PATH} が既に存在します。上書きします。")
        os.remove(SAVE_PATH)

    print(f"保存先     : {SAVE_PATH}")
    print(f"A-1        : RL actor ({ACTOR_PATH})")
    print(f"A-2        : MultiPrefAgent ({len(LANDMARK_NAMES)} 選好)")
    print(f"SEQ_LENGTH : {SEQ_LENGTH}")
    print(f"N_TRAIN    : {N_TRAIN}")
    print(f"N_TEST     : {N_TEST}")
    print(f"デバイス   : {DEVICE}")

    # RL actor
    actor = ActorLSTM().to(DEVICE)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
    actor.eval()
    for p in actor.parameters():
        p.requires_grad = False

    # 環境（世界設定のみ利用。エージェント行動は上書き）
    env_config  = load_config(ENV_CONFIG_PATH)
    env = creator.create_environment(env_config.environment)
    env.init()
    env.off_display()

    # MultiPrefAgent
    multi_agent = MultiPrefAgent()

    h5_file = h5py.File(SAVE_PATH, 'w')
    h5_file.attrs['description'] = (
        'Phase 2 dataset: A-1=RL actor, A-2=MultiPrefAgent (4 preferences)')

    print(f"\n=== train ({N_TRAIN} ep) ===")
    collect_split(env, actor, multi_agent, h5_file, 'train', N_TRAIN)

    print(f"\n=== test ({N_TEST} ep) ===")
    collect_split(env, actor, multi_agent, h5_file, 'test', N_TEST)

    h5_file.close()
    print(f"\n完了: {SAVE_PATH}")
    size_mb = os.path.getsize(SAVE_PATH) / (1024 ** 2)
    print(f"ファイルサイズ: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
