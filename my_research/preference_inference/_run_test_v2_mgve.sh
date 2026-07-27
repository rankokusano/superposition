#!/bin/bash
set -e
cd /work/my_research/preference_inference
for DATA in v2_fixed_red v2_fixed_green v2_fixed_blue v2_fixed_yellow v2_self_rl_other_random_landmark; do
    echo "[v2_exp_b_mg_ve] $DATA"
    python -u test.py \
        --exp_config v2_exp_b_mg_ve --seed 0 --device cuda:1 \
        --data_load_memory --test_epoch 200 \
        --test_data_name $DATA --test_batch_size 10 \
        --test_modes eval \
        --save_targets self_motion self_position other_motion other_position state a2_target_landmark q2_hat
done
echo "MGVE DONE"
