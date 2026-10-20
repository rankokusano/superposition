#!/bin/bash
# v12 stage 2 (docs/v12_experiment_log.md Sec.28): 4 runs of one pair: setting 1 on $GPU1, setting 2 on $GPU2.
# Usage (INSIDE the container from the v11 worktree's my_research/preference_inference): bash run_stage2_v12.sh <seed> <gpu1> <gpu2>
set -u
cd "$(dirname "$0")"
s=$1; g1=$2; g2=$3
[ "$g1" = 0 ] || [ "$g2" = 0 ] && { echo "GPU0 is not allowed"; exit 1; }
launch() {
    local st=$1 c=$2 gpu=$3
    local exp=v12_s2_${st}_${c}_s${s}
    mkdir -p data/result/${exp}
    setsid nohup bash -c "CUDA_VISIBLE_DEVICES=${gpu} python -u train_v12.py --exp_config ${exp} --seed ${s} --device cuda:0 \
        --cudnn_deterministic --data_load_memory > data/result/${exp}/train_seed${s}.log 2>&1; echo \"exit \$?\" > data/result/${exp}/done_seed${s}" \
        < /dev/null > /dev/null 2>&1 &
    echo "launched ${exp} on GPU ${gpu} at $(date)" | tee -a data/result/v12/logs/stage2_launch.log
}
launch green random $g1; launch green mixed $g1
launch pref random $g2;  launch pref mixed $g2
