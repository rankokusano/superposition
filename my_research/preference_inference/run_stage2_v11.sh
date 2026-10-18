#!/bin/bash
# v11 stage 2 (docs/v11_experiment_log.md Sec.7.42): 8 runs (pairs s0, s2 x random/mixed x
# setting green(1)/pref(2)) on GPU3-7. Run INSIDE the container from the v11 worktree's
# my_research/preference_inference. GPU0 is never used; GPU1/2 are the s3 stage-1 runs.
set -u
cd "$(dirname "$0")"
launch() {
    local st=$1 c=$2 s=$3 gpu=$4
    local exp=v11_s2_${st}_${c}_s${s}
    mkdir -p data/result/${exp}
    setsid nohup bash -c "CUDA_VISIBLE_DEVICES=${gpu} python -u train_v11.py --exp_config ${exp} --seed ${s} --device cuda:0 \
        --cudnn_deterministic --data_load_memory > data/result/${exp}/train_seed${s}.log 2>&1; echo \"exit \$?\" > data/result/${exp}/done_seed${s}" \
        < /dev/null > /dev/null 2>&1 &
    echo "launched ${exp} on GPU ${gpu}"
}
launch green random 0 3; launch green mixed 0 3
launch pref random 0 4;  launch pref mixed 0 4
launch green random 2 5; launch green mixed 2 5
launch pref random 2 6;  launch pref mixed 2 7
