#!/bin/bash
set -u; cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES; export DEVICE=cuda:4
python test.py --exp_config r3_stay_400 --seed 0 --device cuda:4 \
  --test_epoch 400 --test_data_name r3_stay --test_batch_size 10 --test_modes eval \
  --save_targets state self_position other_position --test_name r3_stay_fig4c \
  --eval_seed 0 --cudnn_deterministic > logs/r3stay_fig4c_test.log 2>&1
echo "[r3stay-fig4c] rc=$?"
