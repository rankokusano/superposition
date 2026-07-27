#!/bin/bash
# Run test.py on v2 evaluation datasets for both v2 models.
#
# Run inside Docker:
#   docker exec kusano_research bash -c "cd /work/my_research/preference_inference && bash run_test_v2.sh [EPOCH] [SEED]"
set -e

PI=/work/my_research/preference_inference
cd $PI

EPOCH=${1:-200}
SEED=${2:-0}

FIXED_DATASETS=(
    v2_fixed_red
    v2_fixed_green
    v2_fixed_blue
    v2_fixed_yellow
)
RANDOM_DATASET=v2_self_rl_other_random_landmark

_run_tests() {
    local EXP=$1
    local DEVICE=$2
    local SAVE_TARGETS=$3
    for DATA in "${FIXED_DATASETS[@]}" $RANDOM_DATASET; do
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

SAVE_B="self_motion self_position other_motion other_position state a2_target_landmark q2_hat"

echo "=== v2 test runs started at: $(date) ==="

_run_tests v2_exp_b_base  cuda:0 "$SAVE_B" > /tmp/test_v2_base.log    2>&1 &
_run_tests v2_exp_b_mg_ve cuda:1 "$SAVE_B" > /tmp/test_v2_mgve.log    2>&1 &

wait
echo ""
echo "=== v2 test runs done at: $(date) ==="
echo "Logs: /tmp/test_v2_{base,mgve}.log"
