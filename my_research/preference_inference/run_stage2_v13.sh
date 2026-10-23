#!/bin/bash
# v13 stage 2 (docs/v13_experiment_log.md Sec.3.6, 12; exploratory): the experienced (80%) runs of one pair,
# setting 1 (green) on $2 and setting 2 (pref) on $3. The no-experience stage-2 runs are the v12 ones (reused).
# Usage (INSIDE the container from my_research/preference_inference): bash run_stage2_v13.sh <seed> <gpu1> <gpu2>
set -u
cd "$(dirname "$0")"
s=$1
launch() {
    local st=$1 gpu=$2
    case $gpu in 1|2|7) ;; *) echo "GPU $gpu is not allowed for v13"; exit 1 ;; esac
    local exp=v13_s2_${st}_mixed_s${s}
    test -e data/result/${exp}/${s} && { echo "data/result/${exp}/${s} exists; not overwriting"; exit 1; }
    mkdir -p data/result/${exp} data/result/v13/logs
    setsid nohup bash -c "CUDA_VISIBLE_DEVICES=${gpu} python -u train_v12.py --exp_config ${exp} --seed ${s} --device cuda:0 \
        --cudnn_deterministic --data_load_memory > data/result/${exp}/train_seed${s}.log 2>&1; echo \"exit \$?\" > data/result/${exp}/done_seed${s}" \
        < /dev/null > /dev/null 2>&1 &
    echo "launched ${exp} on GPU ${gpu} at $(date)" | tee -a data/result/v13/logs/stage2_launch.log
}
launch green $2
launch pref $3
