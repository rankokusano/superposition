#!/bin/bash
# Analysis for Approach B (base and MGVE)
# Run inside Docker container (kusano_research)
# Usage: bash run_analysis_b.sh [EPOCH] [SEED]

set -e
WORK=/work
PI=$WORK/my_research/preference_inference
cd $PI

EPOCH=${1:-200}
SEED=${2:-0}
DEVICE=${DEVICE:-cuda:0}
DATA_NAME=new_self_rl_other_cycler

for EXP_CONFIG in new_exp_b_base new_exp_b_mg_ve; do
    echo "=== Approach B: $EXP_CONFIG ==="

    echo "--- Generating saved.h5 (epoch=$EPOCH) ---"
    python test.py \
        --exp_config $EXP_CONFIG \
        --seed $SEED \
        --device $DEVICE \
        --data_load_memory \
        --test_epoch $EPOCH \
        --test_data_name $DATA_NAME \
        --test_batch_size 10 \
        --test_modes eval test \
        --save_targets self_motion self_position other_motion other_position state a2_target_landmark q2_hat

    RESULT_DIR=data/result/${EXP_CONFIG}/${SEED}
    SAVED_H5=$RESULT_DIR/test/$DATA_NAME/save

    cd analyze

    echo "--- Plot state: other layer by A-2 landmark ---"
    python plot_state.py \
        --result_dir ../$SAVED_H5 \
        --epoch $EPOCH \
        --mode eval \
        --plot a2_landmark \
        --plot_layer other \
        --analyze_layer other

    echo "--- Regression: A-2 landmark from hidden states ---"
    for LAYER in self other; do
        python regression.py \
            --result_dir ../$SAVED_H5 \
            --epoch $EPOCH \
            --mode eval \
            --layer $LAYER
    done

    cd ..
done

echo "=== Q-map analysis ==="
python analyze/plot_q_map.py \
    --h5 data/data/new_grid_q_map/data.h5 \
    --savedir data/result/q_map \
    --num_div 20

echo "=== Approach B analysis done ==="
