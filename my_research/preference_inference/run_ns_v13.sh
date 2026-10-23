#!/bin/bash
# v13-NS stage 1 (docs/v13_experiment_log.md Sec.19-21): per seed, the no-experience and experienced (80%) runs
# on one GPU, each mse 200ep -> l1 200ep (Decoder only, pretrain = same seed's mse ep200), via train_v13ns.py.
# Usage (INSIDE the container from my_research/preference_inference): bash run_ns_v13.sh <seed> <gpu>
# GPUs allowed for v13-NS: 6, 7, 2 (Sec.20). GPU0 and GPU3-5 are never used.
set -u
cd "$(dirname "$0")"
seed=$1; gpu=$2
case $gpu in 2|6|7) ;; *) echo "GPU $gpu is not allowed for v13-NS"; exit 1 ;; esac
for cond in random mixed; do
    mse=v13ns_s1_${cond}_mse l1=v13ns_s1_${cond}_l1_s${seed}
    test -e data/result/${mse}/${seed} && { echo "data/result/${mse}/${seed} exists; not overwriting"; exit 1; }
    mkdir -p data/result/${mse} data/result/${l1}
    setsid nohup bash -c "
        CUDA_VISIBLE_DEVICES=${gpu} python -u train_v13ns.py --exp_config ${mse} --seed ${seed} --device cuda:0 \
            --cudnn_deterministic --data_load_memory > data/result/${mse}/train_seed${seed}.log 2>&1 &&
        test -f data/result/${mse}/${seed}/model/00200.pth &&
        CUDA_VISIBLE_DEVICES=${gpu} python -u train_v13ns.py --exp_config ${l1} --seed ${seed} --device cuda:0 \
            --cudnn_deterministic --data_load_memory > data/result/${l1}/train_seed${seed}.log 2>&1
        echo \"exit \$?\" > data/result/${l1}/chain_seed${seed}.done
    " < /dev/null > /dev/null 2>&1 &
    echo "launched NS ${cond} seed ${seed} on GPU ${gpu} at $(date)"
done
