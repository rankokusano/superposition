#!/bin/bash
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:6
# v5_base_mse late-5 (ep360-400) = the stage where encoders+SM actually trained.
# v5_base_l1 4-axis is degenerate (decoder-only freeze).
bash analyze/run_eval_lateckpt.sh v5_base_mse 400 0 r3_stay > logs/v5_mse_eval_r3stay.log 2>&1
echo "[v5-mse-eval] r3_stay rc=$?"
bash analyze/run_eval_lateckpt.sh v5_base_mse 400 0 r2_a1random_a2rl > logs/v5_mse_eval_common.log 2>&1
echo "[v5-mse-eval] common rc=$?"
python analyze/check_convergence.py v5_base_mse --seeds 0 2>&1 | grep -v "^saved:"
echo "[v5-mse-eval] DONE $(date -u)"
