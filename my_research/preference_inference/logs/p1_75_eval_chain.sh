#!/bin/bash
set -u
cd /work/my_research/preference_inference
running(){ for p in /proc/[0-9]*; do [ -r "$p/cmdline" ] || continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -q "exp_config r4_move_sym" && return 0; done; return 1; }
echo "[eval-chain] waiting for training to finish ..."
while running; do sleep 60; done
echo "[eval-chain] training done. ckpts:"; ls data/result/r4_move_sym/0/model/ | tail -3
export DEVICE=cuda:0
export CUDA_VISIBLE_DEVICES=7
echo "[eval-chain] late-5 harness (r4_move_sym, E=400, dataset r2_a1red_a2red) ..."
bash analyze/run_eval_lateckpt.sh r4_move_sym 400 0 r2_a1red_a2red 2>&1
echo "[eval-chain] r4_ve_eval with A-1 critic (true Q on A-1 scale) at last 5 ckpts ..."
for ep in 360 370 380 390 400; do
  python analyze/r4_ve_eval.py --exp_config r4_move_sym --epoch $ep --seed 0 --eval_seed 0 \
    --a2_critic_path data/model/v3_rl_critic.pth \
    --data_h5 data/data/r2_a1red_a2red/data.h5 \
    --label r4_move_sym_a1critic_ep${ep} 2>&1 | tail -20
done
echo "[eval-chain] DONE"
