#!/bin/bash
# v12 trial (docs/v12_experiment_log.md Sec.4): v12_trial_{random,mixed}_mse, seed 0, 150 ep, both on GPU7.
# Run INSIDE the container from the v11 worktree's my_research/preference_inference. GPU0 is never used.
set -u
cd "$(dirname "$0")"
for c in random mixed; do
    exp=v12_trial_${c}_mse
    mkdir -p data/result/${exp}
    setsid nohup bash -c "CUDA_VISIBLE_DEVICES=7 python -u train_v12.py --exp_config ${exp} --seed 0 --device cuda:0 \
        --cudnn_deterministic --data_load_memory > data/result/${exp}/train_seed0.log 2>&1; echo \"exit \$?\" > data/result/${exp}/done_seed0" \
        < /dev/null > /dev/null 2>&1 &
    echo "launched ${exp} on GPU 7"
done
