#!/bin/bash
# v14-C stage 1 (docs/v14_experiment_log.md Sec.5, 10, 13). Per seed one chain: mse 200ep -> l1 200ep (Decoder only,
# pretrain = same seed's mse ep200), with train_v14.py (signed pleasure, loader with a1_liked_t).
# Run INSIDE the container from the v11 worktree's my_research/preference_inference:
#     bash run_main_v14c.sh <seed> <gpu>
# GPU5/6 only, unless ALLOW_GPU=<gpu> is set for a GPU Kusano-san approved (log entry first).
set -u
cd "$(dirname "$0")"
seed=$1 gpu=$2
case $gpu in 5|6) ;; *) [ "${ALLOW_GPU:-}" = "$gpu" ] || { echo "GPU $gpu is not allowed for v14"; exit 1; } ;; esac
mse=v14_s1_mixedpref_mse l1=v14_s1_mixedpref_l1_s${seed}
test -e data/result/${mse}/${seed} && { echo "data/result/${mse}/${seed} exists; not overwriting"; exit 1; }
mkdir -p data/result/${mse} data/result/${l1}
setsid nohup bash -c "
    CUDA_VISIBLE_DEVICES=${gpu} python -u train_v14.py --exp_config ${mse} --seed ${seed} --device cuda:0 \
        --cudnn_deterministic --data_load_memory > data/result/${mse}/train_seed${seed}.log 2>&1 &&
    test -f data/result/${mse}/${seed}/model/00200.pth &&
    CUDA_VISIBLE_DEVICES=${gpu} python -u train_v14.py --exp_config ${l1} --seed ${seed} --device cuda:0 \
        --cudnn_deterministic --data_load_memory > data/result/${l1}/train_seed${seed}.log 2>&1
    echo \"exit \$?\" > data/result/${l1}/chain_seed${seed}.done
" < /dev/null > /dev/null 2>&1 &
echo "launched v14-C seed ${seed} on GPU ${gpu}"
