#!/bin/bash
set -u; cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES; export DEVICE=cuda:6
# P0-4: r2_a1random_a2rl, A-2 green critic, ep400
python analyze/r4_ve_eval.py --exp_config r4_move_exp3parity --epoch 400 --seed 0 --eval_seed 0 \
  --a2_critic_path data/model/v3_rl_a2_critic.pth --data_h5 data/data/r2_a1random_a2rl/data.h5 \
  --target_pos=-9,-9 --dump_arrays --label f4_p0_4 2>&1 | grep -E "Saved|overall|true Q2 std|true_q2_std" | grep -vE "it/s"
# P1.75: r2_a1red_a2red, A-1 critic (symmetric), ep400
python analyze/r4_ve_eval.py --exp_config r4_move_sym --epoch 400 --seed 0 --eval_seed 0 \
  --a2_critic_path data/model/v3_rl_critic.pth --data_h5 data/data/r2_a1red_a2red/data.h5 \
  --target_pos=-9,9 --dump_arrays --label f4_p1_75 2>&1 | grep -E "Saved|overall|true Q2 std|true_q2_std" | grep -vE "it/s"
# P2-a: r2_a1random_a2rl, A-2 green critic, ep400
python analyze/r4_ve_eval.py --exp_config v5_r4_move --epoch 400 --seed 0 --eval_seed 0 \
  --a2_critic_path data/model/v3_rl_a2_critic.pth --data_h5 data/data/r2_a1random_a2rl/data.h5 \
  --target_pos=-9,-9 --dump_arrays --label f4_p2_a 2>&1 | grep -E "Saved|overall|true Q2 std|true_q2_std" | grep -vE "it/s"
echo "[f4-arrays] DONE"
