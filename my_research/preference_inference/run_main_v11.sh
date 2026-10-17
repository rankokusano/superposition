#!/bin/bash
# v11 stage 1 main run (docs/v11_experiment_log.md Sec.5.3, 7.13, 7.16).
# 2 conditions x 3 seeds; per (condition, seed) one GPU runs the paper's exp1
# recipe in sequence: mse stage 200ep (all modules from scratch) -> l1 stage
# 200ep (Decoder only, pretrain = same seed's mse ep200).
# Run INSIDE the container from the v11 worktree's my_research/preference_inference:
#     bash run_main_v11.sh
# GPU map: random s0,s1,s2 -> GPU1,2,3 ; mixed s0,s1,s2 -> GPU4,5,6. GPU0 is never used.
set -u
cd "$(dirname "$0")"
launch() {
    local cond=$1 seed=$2 gpu=$3
    local mse=v11_s1_${cond}_mse l1=v11_s1_${cond}_l1_s${seed}
    mkdir -p data/result/${mse} data/result/${l1}
    setsid nohup bash -c "
        CUDA_VISIBLE_DEVICES=${gpu} python -u train_v11.py --exp_config ${mse} --seed ${seed} --device cuda:0 \
            --cudnn_deterministic --data_load_memory > data/result/${mse}/train_seed${seed}.log 2>&1 &&
        test -f data/result/${mse}/${seed}/model/00200.pth &&
        CUDA_VISIBLE_DEVICES=${gpu} python -u train_v11.py --exp_config ${l1} --seed ${seed} --device cuda:0 \
            --cudnn_deterministic --data_load_memory > data/result/${l1}/train_seed${seed}.log 2>&1
        echo \"exit \$?\" > data/result/${l1}/chain_seed${seed}.done
    " < /dev/null > /dev/null 2>&1 &
    echo "launched ${cond} seed ${seed} on GPU ${gpu}"
}
launch random 0 1
launch random 1 2
launch random 2 3
launch mixed 0 4
launch mixed 1 5
launch mixed 2 6
