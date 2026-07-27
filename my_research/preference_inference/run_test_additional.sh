#!/bin/bash
# Run saved.h5 generation for additional validation datasets using existing trained models.
# No re-training needed. Runs 3 experiments in parallel on 3 GPUs.
# Usage: bash run_test_additional.sh [EPOCH] [SEED]
set -e

PI=/work/my_research/preference_inference
cd $PI

EPOCH=${1:-200}
SEED=${2:-0}

DATASETS=(
    test_self_rl_other_random
    test_self_rl_other_fixed_red
    test_self_rl_other_fixed_green
    test_self_rl_other_fixed_blue
    test_self_rl_other_fixed_yellow
)

_run_tests() {
    local EXP=$1
    local DEVICE=$2
    local SAVE_TARGETS=$3
    for DATA in "${DATASETS[@]}"; do
        echo "  [$EXP / $DEVICE] $DATA"
        python test.py \
            --exp_config $EXP \
            --seed $SEED \
            --device $DEVICE \
            --data_load_memory \
            --test_epoch $EPOCH \
            --test_data_name $DATA \
            --test_batch_size 10 \
            --test_modes eval \
            --save_targets $SAVE_TARGETS
    done
}

SAVE_A="self_motion self_position other_motion other_position state a2_target_landmark"
SAVE_B="self_motion self_position other_motion other_position state a2_target_landmark q2_hat"

echo "=== Running tests for all 3 experiments in parallel ==="

_run_tests new_exp_a       cuda:0 "$SAVE_A" > /tmp/test_add_a.log    2>&1 &
_run_tests new_exp_b_base  cuda:1 "$SAVE_B" > /tmp/test_add_b.log    2>&1 &
_run_tests new_exp_b_mg_ve cuda:2 "$SAVE_B" > /tmp/test_add_mgve.log 2>&1 &

wait
echo "=== All additional test runs done ==="
echo "Logs: /tmp/test_add_{a,b,mgve}.log"
