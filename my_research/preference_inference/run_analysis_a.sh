#!/bin/bash
# Analysis for Approach A: SuperpositionNetworkFeaturePrediction
# Run inside Docker container (kusano_research)
# Usage: bash run_analysis_a.sh [EPOCH] [SEED]

set -e
WORK=/work
PI=$WORK/my_research/preference_inference
cd $PI

EPOCH=${1:-200}
SEED=${2:-0}
DEVICE=${DEVICE:-cuda:0}
EXP_CONFIG=new_exp_a
DATA_NAME=new_self_rl_other_cycler

# Generate saved.h5 via test.py
echo "=== Generating saved.h5 for Approach A (epoch=$EPOCH) ==="
python test.py \
    --exp_config $EXP_CONFIG \
    --seed $SEED \
    --device $DEVICE \
    --data_load_memory \
    --test_epoch $EPOCH \
    --test_data_name $DATA_NAME \
    --test_batch_size 10 \
    --test_modes eval test \
    --save_targets self_motion self_position other_motion other_position state a2_target_landmark

# saved.h5 is at: data/result/{exp_config}/{seed}/test/{data_name}/save/saved.h5
RESULT_DIR=data/result/${EXP_CONFIG}/${SEED}
SAVED_H5=$RESULT_DIR/test/$DATA_NAME/save

cd analyze

echo "=== Plot state by self_position (self layer) ==="
python plot_state.py \
    --result_dir ../$SAVED_H5 \
    --epoch $EPOCH \
    --mode eval \
    --plot self_position \
    --plot_layer self \
    --analyze_layer self

echo "=== Plot state by other_position (other layer) ==="
python plot_state.py \
    --result_dir ../$SAVED_H5 \
    --epoch $EPOCH \
    --mode eval \
    --plot other_position \
    --plot_layer other \
    --analyze_layer other

echo "=== Plot state by A-2 landmark (other layer) ==="
python plot_state.py \
    --result_dir ../$SAVED_H5 \
    --epoch $EPOCH \
    --mode eval \
    --plot a2_landmark \
    --plot_layer other \
    --analyze_layer other

echo "=== Regression: decode A-2 landmark from hidden states ==="
for LAYER in self other; do
    python regression.py \
        --result_dir ../$SAVED_H5 \
        --epoch $EPOCH \
        --mode eval \
        --layer $LAYER
done

echo "=== Approach A analysis done ==="
cd ..
