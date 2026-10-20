#!/bin/bash
# v12 spare pair s3, stage 1 (docs/v12_experiment_log.md Sec.26): waits until the early G1-G3 evaluation on
# GPU1/2 is over, then random s3 on GPU1 and mixed s3 on GPU2 (mse 200ep -> l1 200ep, as run_main_v12.sh).
# Run INSIDE the container from the v11 worktree's my_research/preference_inference. GPU0 never used.
set -u
cd "$(dirname "$0")"
until [ -f data/result/v12/logs/eval_s1_early/gpu1.done ] && [ -f data/result/v12/logs/eval_s1_early/gpu2.done ]; do sleep 60; done
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
    echo "launched ${cond} seed ${seed} on GPU ${gpu} at $(date)" >> data/result/v12/logs/s3_launch.log
}
launch random 3 1
launch mixed 3 2
