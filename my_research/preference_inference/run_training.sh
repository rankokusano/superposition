#!/bin/bash
# Run inside Docker container (kusano_research)
# Usage: bash run_training.sh [SEED] [DEVICE]
# Trains all 3 experiment configurations.

set -e
WORK=/work
PI=$WORK/my_research/preference_inference
cd $PI

SEED=${1:-0}
DEVICE=${2:-cuda:0}

echo "=== Training Approach A: SuperpositionNetworkFeaturePrediction ==="
python train.py \
    --exp_config new_exp_a \
    --seed $SEED \
    --device $DEVICE \
    --data_load_memory

echo "=== Training Approach B-base: SuperpositionNetworkApproachBBase ==="
python train.py \
    --exp_config new_exp_b_base \
    --seed $SEED \
    --device $DEVICE \
    --data_load_memory

echo "=== Training Approach B-MGVE: SuperpositionNetworkApproachBMGVE ==="
python train.py \
    --exp_config new_exp_b_mg_ve \
    --seed $SEED \
    --device $DEVICE \
    --data_load_memory

echo "=== All training done ==="
