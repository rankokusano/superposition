#!/bin/bash
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:5
# R3-stay_400 (v4, frozen Encoder) on the COMMON protocol dataset, to compare
# apples-to-apples with v5_base_mse @ r2_a1random_a2rl. 3 seeds.
for s in 0 1 2; do
  echo "###### r3_stay_400 s$s @ r2_a1random_a2rl ######"
  bash analyze/run_eval_lateckpt.sh r3_stay_400 400 $s r2_a1random_a2rl > logs/r3stay_common_s${s}.log 2>&1
  echo "  rc=$?"
done
echo "[r3stay-common] DONE $(date -u)"
