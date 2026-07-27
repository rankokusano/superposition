#!/bin/bash
# Train v2 Superposition models (B-base and B-MGVE) using v2 data.
# B-base trains first, then B-MGVE uses it as pretrain.
#
# Run inside Docker:
#   docker exec kusano_research bash -c "cd /work/my_research/preference_inference && bash run_training_v2.sh [SEED]"
set -e

PI=/work/my_research/preference_inference
cd $PI

SEED=${1:-0}

echo "=== v2 training started at: $(date) ==="

# Step 1: Train v2_exp_b_base on v2_self_rl_other_stay
echo ""
echo "--- Training v2_exp_b_base (A-2 stays) ---"
python train.py \
    --exp_config v2_exp_b_base \
    --seed $SEED \
    --device cuda:0 \
    --data_load_memory 2>&1 | tee /tmp/train_v2_b_base.log

echo ""
echo "--- v2_exp_b_base training done ---"

# Step 2: Train v2_exp_b_mg_ve on v2_self_rl_other_random_landmark (pretrain from B-base)
echo ""
echo "--- Training v2_exp_b_mg_ve (A-2 random landmark, pretrain from B-base) ---"
python train.py \
    --exp_config v2_exp_b_mg_ve \
    --seed $SEED \
    --device cuda:0 \
    --data_load_memory 2>&1 | tee /tmp/train_v2_b_mgve.log

echo ""
echo "=== v2 training done at: $(date) ==="
echo "Logs: /tmp/train_v2_b_{base,mgve}.log"
