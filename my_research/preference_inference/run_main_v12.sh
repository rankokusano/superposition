#!/bin/bash
# v12 stage 1 main run (docs/v12_experiment_log.md Sec.11, 13; instructions Sec.4.2). NOT launched before Kusano-san confirms.
# 2 conditions x 3 seeds; per (condition, seed) one GPU runs the paper's exp1
# recipe in sequence: mse stage 200ep (all modules from scratch) -> l1 stage
# 200ep (Decoder only, pretrain = same seed's mse ep200).
# Run INSIDE the container from the v11 worktree's my_research/preference_inference:
#     bash run_main_v11.sh
# GPU map (Sec.11.4, agreed): random s0,s1,s2 -> GPU3,4,5 ; mixed s0,s1 -> GPU6,7 ; mixed s2 -> GPU7 (shared). GPU0 never used.
set -u
cd "$(dirname "$0")"
launch() {
    local cond=$1 seed=$2 gpu=$3
    local mse=v12_s1_${cond}_mse l1=v12_s1_${cond}_l1_s${seed}
    mkdir -p data/result/${mse} data/result/${l1}
    setsid nohup bash -c "
        CUDA_VISIBLE_DEVICES=${gpu} python -u train_v12.py --exp_config ${mse} --seed ${seed} --device cuda:0 \
            --cudnn_deterministic --data_load_memory > data/result/${mse}/train_seed${seed}.log 2>&1 &&
        test -f data/result/${mse}/${seed}/model/00200.pth &&
        CUDA_VISIBLE_DEVICES=${gpu} python -u train_v12.py --exp_config ${l1} --seed ${seed} --device cuda:0 \
            --cudnn_deterministic --data_load_memory > data/result/${l1}/train_seed${seed}.log 2>&1
        echo \"exit \$?\" > data/result/${l1}/chain_seed${seed}.done
    " < /dev/null > /dev/null 2>&1 &
    echo "launched ${cond} seed ${seed} on GPU ${gpu}"
}
launch random 0 3
launch random 1 4
launch random 2 5
launch mixed 0 6
launch mixed 1 7
launch mixed 2 7
