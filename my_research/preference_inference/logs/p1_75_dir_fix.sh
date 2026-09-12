#!/bin/bash
set -u
cd /work/my_research/preference_inference
export CUDA_VISIBLE_DEVICES=7
for ep in 360 370 380 390 400; do
  echo "########## r4_move_sym RED-target ep${ep} ##########"
  python analyze/r4_ve_eval.py --exp_config r4_move_sym --epoch $ep --seed 0 --eval_seed 0 \
    --a2_critic_path data/model/v3_rl_critic.pth \
    --data_h5 data/data/r2_a1red_a2red/data.h5 \
    --target_pos=-9,9 \
    --label r4_move_sym_red_ep${ep} 2>&1 | grep -vE "it/s\]|it\]" | grep -E "Loading|cos_sim|angular|within 45|true Q2 std|overall:|Q.2 ->|A-1 self|A-2 other|Saved:"
done
echo "[dir-fix] DONE"
