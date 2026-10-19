#!/bin/bash
# v12 trials (P-c) (docs/v12_experiment_log.md Sec.10): P-c random on GPU3, P-c mixed on GPU4;
# seed 0, 150 ep. Run INSIDE the container from the v11 worktree's
# my_research/preference_inference. GPU0 is never used; GPU1/2 stay with v11 s3.
set -u
cd "$(dirname "$0")"
launch() {
    local exp=$1 gpu=$2
    mkdir -p data/result/${exp}
    setsid nohup bash -c "CUDA_VISIBLE_DEVICES=${gpu} python -u train_v12.py --exp_config ${exp} --seed 0 --device cuda:0 \
        --cudnn_deterministic --data_load_memory > data/result/${exp}/train_seed0.log 2>&1; echo \"exit \$?\" > data/result/${exp}/done_seed0" \
        < /dev/null > /dev/null 2>&1 &
    echo "launched ${exp} on GPU ${gpu}"
}
launch v12_trial_pc_random_mse 3
launch v12_trial_pc_mixed_mse 4
