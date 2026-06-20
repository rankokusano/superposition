"""
collect_data_slow_pref.py

A-2 の静止問題を解消した多選好データ収集。
speed=0.1 にすることで 100 step 以内にランドマークへ到達しないケースが大半となり、
エピソード全体で A-2 が動き続ける。

【速度設計】
    speed=0.1 → 100 step で最大 10 unit 移動
    フィールド対角 ≈ 25.5 unit → ランダムスタートの約 85% が到達しない

【出力】
    data/data/self_random_other_stay_slow_pref/data.h5
    ← 既存データは一切変更しない

実行コマンド（コンテナ内）:
    cd /work
    DISPLAY=:99 MESA_GL_VERSION_OVERRIDE=3.3 \\
    python3 my_research/step3/collect_data_slow_pref.py \\
    2>&1 | tee my_research/step3/results/collect_slow_pref.log
"""

import sys, os

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
from agent2_multi_pref import MultiPrefAgent

# =========================================================================
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORLD_HALF  = 9.5
SEQ_LENGTH  = 101
N_TRAIN     = 2100
N_TEST      = 220
A2_SPEED    = 0.1   # 静止問題を解消するための低速設定

ACTOR_PATH      = os.path.join(MY_RESEARCH, 'rl_model_v6_actor.pth')
ENV_CONFIG_PATH = os.path.join(PROJ_ROOT, 'simulation/config/collect/self_random_other_stay.yml')
SAVE_DIR        = os.path.join(PROJ_ROOT, 'data/data/self_random_other_stay_slow_pref')
SAVE_PATH       = os.path.join(SAVE_DIR, 'data.h5')

# =========================================================================

def collect_split(env, actor, agent2, h5_file, mode, n_episodes):
    H = env.world.config.camera.agent.height
    W = env.world.config.camera.agent.width

    grp   = h5_file.require_group(mode)
    ds_sv = grp.create_dataset('self_vision',    (n_episodes, SEQ_LENGTH, H, W, 3), dtype='f')
    ds_sm = grp.create_dataset('self_motion',    (n_episodes, SEQ_LENGTH, 2),       dtype='f')
    ds_sp = grp.create_dataset('self_position',  (n_episodes, SEQ_LENGTH, 2),       dtype='f')
    ds_om = grp.create_dataset('other_motion',   (n_episodes, SEQ_LENGTH, 2),       dtype='f')
    ds_op = grp.create_dataset('other_position', (n_episodes, SEQ_LENGTH, 2),       dtype='f')

    n_arrived = 0   # ランドマーク到達エピソード数の追跡

    for n in range(n_episodes):
        env.reset()
        pref_name = agent2.reset()
        actor_h = None
        sp = env.self_agent.p.copy()
        op = env.other_agent.p.copy()
        ep_arrived = False

        for t in range(SEQ_LENGTH):
            v = env.capture()
            v_tensor = torch.FloatTensor(v.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                sm_tensor, _, actor_h = actor.sample(v_tensor, actor_h)
            sm = sm_tensor.squeeze(0).cpu().numpy()
            om = agent2.get_action(op)

            ds_sv[n, t] = v
            ds_sm[n, t] = sm
            ds_sp[n, t] = sp
            ds_om[n, t] = om
            ds_op[n, t] = op

            # 到達判定（デバッグ用）
            if np.linalg.norm(om) < 1e-6 and t > 0 and not ep_arrived:
                ep_arrived = True
                n_arrived += 1

            sp = np.clip(sp + sm, -WORLD_HALF, WORLD_HALF)
            op = np.clip(op + om, -WORLD_HALF, WORLD_HALF)
            env.self_agent.p  = sp.copy()
            env.other_agent.p = op.copy()

        if (n + 1) % 200 == 0:
            print(f"  [{mode}] {n+1}/{n_episodes}  (到達率: {n_arrived/(n+1)*100:.1f}%)")

    print(f"  [{mode}] 完了: {n_episodes} ep  到達エピソード: {n_arrived} ({n_arrived/n_episodes*100:.1f}%)")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    if os.path.exists(SAVE_PATH):
        print(f"警告: {SAVE_PATH} が既に存在します。上書きします。")
        os.remove(SAVE_PATH)

    print(f"保存先     : {SAVE_PATH}")
    print(f"A-1        : RL actor")
    print(f"A-2        : MultiPrefAgent(speed={A2_SPEED})  ← 静止問題解消版")
    print(f"SEQ_LENGTH : {SEQ_LENGTH}")
    print(f"N_TRAIN    : {N_TRAIN}  N_TEST: {N_TEST}")

    actor = ActorLSTM().to(DEVICE)
    actor.load_state_dict(torch.load(ACTOR_PATH, map_location=DEVICE))
    actor.eval()
    for p in actor.parameters():
        p.requires_grad = False

    env_config = load_config(ENV_CONFIG_PATH)
    env = creator.create_environment(env_config.environment)
    env.init()
    env.off_display()

    agent2 = MultiPrefAgent(speed=A2_SPEED)

    h5_file = h5py.File(SAVE_PATH, 'w')
    h5_file.attrs['description'] = f'Multi pref slow: A-2 speed={A2_SPEED}'

    print(f"\n=== train ({N_TRAIN} ep) ===")
    collect_split(env, actor, agent2, h5_file, 'train', N_TRAIN)

    print(f"\n=== test ({N_TEST} ep) ===")
    collect_split(env, actor, agent2, h5_file, 'test', N_TEST)

    h5_file.close()
    print(f"\n完了: {SAVE_PATH}")
    print(f"ファイルサイズ: {os.path.getsize(SAVE_PATH)/(1024**2):.1f} MB")


if __name__ == '__main__':
    main()
