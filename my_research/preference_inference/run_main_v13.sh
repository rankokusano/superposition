#!/bin/bash
# v13 stage 1 main run (docs/v13_experiment_log.md Sec.3.2, 4). NOT launched before Kusano-san confirms.
# Experienced (80%) only; the no-experience runs are the v12 ones (Sec.1). Per seed one GPU runs
# mse 200ep -> l1 200ep (Decoder only, pretrain = same seed's mse ep200).
# Run INSIDE the container from the v11 worktree's my_research/preference_inference:
#     bash run_main_v13.sh            # s0,s1,s2 on GPU7,1,2
#     bash run_main_v13.sh 3 <gpu>    # spare pair s3
# GPU0 and GPU3-6 are never used.
set -u
cd "$(dirname "$0")"
launch() {
    local seed=$1 gpu=$2
    case $gpu in 1|2|7) ;; *) echo "GPU $gpu is not allowed for v13"; exit 1 ;; esac
    local mse=v13_s1_mixed_mse l1=v13_s1_mixed_l1_s${seed}
    test -e data/result/${mse}/${seed} && { echo "data/result/${mse}/${seed} exists; not overwriting"; exit 1; }
    mkdir -p data/result/${mse} data/result/${l1}
    setsid nohup bash -c "
        CUDA_VISIBLE_DEVICES=${gpu} python -u train_v12.py --exp_config ${mse} --seed ${seed} --device cuda:0 \
            --cudnn_deterministic --data_load_memory > data/result/${mse}/train_seed${seed}.log 2>&1 &&
        test -f data/result/${mse}/${seed}/model/00200.pth &&
        CUDA_VISIBLE_DEVICES=${gpu} python -u train_v12.py --exp_config ${l1} --seed ${seed} --device cuda:0 \
            --cudnn_deterministic --data_load_memory > data/result/${l1}/train_seed${seed}.log 2>&1
        echo \"exit \$?\" > data/result/${l1}/chain_seed${seed}.done
    " < /dev/null > /dev/null 2>&1 &
    echo "launched mixed80 seed ${seed} on GPU ${gpu}"
}
if [ $# -eq 2 ]; then
    launch $1 $2
else
    launch 0 7
    launch 1 1
    launch 2 2
fi
