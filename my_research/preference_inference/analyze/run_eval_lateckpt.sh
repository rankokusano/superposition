#!/bin/bash
# Late-checkpoint evaluation (research_plan_v4_v5.md Sec.P0-5, finalised
# 2026-09-06 per user direction: option (b) applied UNIFORMLY to all runs).
#
# Runs the canonical P0-1 harness (run_eval_v4v5.sh) at the LAST 5 saved
# checkpoints (E, E-10, ..., E-40) and lets aggregate_lateckpt.py compute
# mean +/- sd across those 5. The sd here is INTER-CHECKPOINT variation
# (training oscillation) -- a different quantity from the --eval_seed sd
# (masking noise). Selecting checkpoints by position (last 5), not by train
# loss, so there is no cherry-picking.
#
# Usage (from /work/my_research/preference_inference):
#   DEVICE=cuda:7 bash analyze/run_eval_lateckpt.sh <exp_config> <E> [train_seed] [dataset]
set -e
EXP="${1:?exp_config}"
E="${2:?last epoch}"
SEED="${3:-0}"
DATASET="${4:-r2_a1random_a2rl}"

for k in 0 10 20 30 40; do
  EP=$((E - k))
  echo "########## $EXP seed$SEED  epoch $EP ##########"
  DEVICE="${DEVICE:-cuda:0}" bash analyze/run_eval_v4v5.sh "$EXP" "$EP" 0 "$DATASET" "$SEED"
done

python analyze/aggregate_lateckpt.py "$EXP" "$E" --seed "$SEED" --dataset "$DATASET"
