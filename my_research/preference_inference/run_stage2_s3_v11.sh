#!/bin/bash
# v11 stage 2 for the spare pair s3 (docs/v11_experiment_log.md Sec.7.56): 4 runs, setting 1 on GPU1,
# setting 2 on GPU2. Run INSIDE the container from the v11 worktree's my_research/preference_inference.
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
launch green random 3 1; launch green mixed 3 1
launch pref random 3 2;  launch pref mixed 3 2
