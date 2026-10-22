#!/bin/bash
# v14-B one-bit stage 2 (docs/v14_experiment_log.md Sec.4, 10, 14). Data (2) v11_s2_pref, 200 epochs, G only.
# Run INSIDE the container from the v11 worktree's my_research/preference_inference:
#     bash run_stage2u_v14.sh <gpu> <cond:seed> ...     e.g. bash run_stage2u_v14.sh 5 mixed:0 random:0
# GPU5 / GPU6 only. Never overwrites an existing run directory. End marker data/result/<exp>/done_seed<s>.
set -u
cd "$(dirname "$0")"
gpu=$1; shift
case $gpu in 5|6) ;; *) echo "GPU $gpu is not allowed for v14"; exit 1 ;; esac
for cs in "$@"; do
    c=${cs%%:*}; s=${cs##*:}
    exp=v14_s2u_pref_${c}_s${s}
    test -e data/result/${exp}/${s} && { echo "data/result/${exp}/${s} exists; not overwriting"; continue; }
    mkdir -p data/result/${exp}
    setsid nohup bash -c "
        CUDA_VISIBLE_DEVICES=${gpu} python -u train_v14.py --exp_config ${exp} --seed ${s} --device cuda:0 \
            --cudnn_deterministic --data_load_memory > data/result/${exp}/train_seed${s}.log 2>&1
        echo \"exit \$?\" > data/result/${exp}/done_seed${s}
    " < /dev/null > /dev/null 2>&1 &
    echo "launched ${exp} seed ${s} on GPU ${gpu}"
done
