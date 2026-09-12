#!/bin/bash
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:7
bash analyze/run_eval_lateckpt.sh v5_base_l1 200 0 r3_stay > logs/v5_eval_l1.log 2>&1
echo "[v5-eval] l1 rc=$?"
# also common-protocol dataset for cross-check
bash analyze/run_eval_lateckpt.sh v5_base_l1 200 0 r2_a1random_a2rl > logs/v5_eval_l1_common.log 2>&1
echo "[v5-eval] common rc=$?"
echo "[v5-eval] DONE $(date -u)"
