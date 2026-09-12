#!/bin/bash
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:5
bash analyze/run_eval_lateckpt.sh r3_stay_400 400 2 r2_a1random_a2rl > logs/r3stay_common_s2.log 2>&1
echo "[r3stay-s2] rc=$? $(date -u)"
