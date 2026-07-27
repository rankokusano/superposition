#!/bin/bash
# Run test.py for v3 B-base and B-MGVE models to save hidden states.
# Run from /work/my_research/preference_inference inside Docker.
set -e

EPOCH=200
SEED=0
TEST_DATA=v3_b_mgve_train
BATCH=10

SAVE="self_position other_position state"

echo "=== [v3_exp_b_base_l1] test on $TEST_DATA ==="
python -u test.py \
    --exp_config v3_exp_b_base_l1 --seed $SEED --device cuda:0 \
    --test_epoch $EPOCH --test_data_name $TEST_DATA \
    --test_batch_size $BATCH --test_modes eval \
    --save_targets $SAVE

echo "=== [v3_exp_b_mgve] test on $TEST_DATA ==="
python -u test.py \
    --exp_config v3_exp_b_mgve --seed $SEED --device cuda:0 \
    --test_epoch $EPOCH --test_data_name $TEST_DATA \
    --test_batch_size $BATCH --test_modes eval \
    --save_targets $SAVE q2_hat

echo "Done."
