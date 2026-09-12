#!/bin/bash
# P1.75 CORRECT r4_ve_eval: A-1 critic (v3_rl_critic.pth) as A-2 true Q,
# symmetric dataset (r2_a1red_a2red), direction target = Red landmark (-9,9).
# Waits for the main eval-chain to finish first (shares GPU7).
set -u
cd /work/my_research/preference_inference
export CUDA_VISIBLE_DEVICES=7
running(){ for p in /proc/[0-9]*; do [ -r "$p/cmdline" ] || continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -q "p1_75_eval_chain.sh" && return 0; done; return 1; }
echo "[correct] waiting for main eval-chain ..."
while running; do sleep 60; done
echo "[correct] $(date -u) running corrected P1.75 r4_ve_eval at last 5 ckpts"
for ep in 360 370 380 390 400; do
  echo "########## r4_move_sym CORRECT ep${ep} ##########"
  python analyze/r4_ve_eval.py --exp_config r4_move_sym --epoch $ep --seed 0 --eval_seed 0 \
    --a2_critic_path data/model/v3_rl_critic.pth \
    --data_h5 data/data/r2_a1red_a2red/data.h5 \
    --target_pos -9,9 \
    --label r4_move_sym_correct_ep${ep} 2>&1 | grep -vE "it/s\]|it\]" | grep -E "Loading|real |zero |true_q2 |constant_mean |cos_sim|true Q2 std|per-dim|overall:|delta|Saved:|a2_critic|Item"
done
echo "[correct] DONE $(date -u)"
