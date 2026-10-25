#!/bin/bash
# Paper extra C training (docs/paper_extra_log.md Sec.4). NOT launched before Kusano-san confirms.
# Per seed one chain on one GPU: mse 200ep -> l1 200ep (Decoder only, pretrain = same seed's mse ep200),
# signed-pleasure model on paper_s1_mixed80_signrand (v13_s1_mixed80 + per-episode sign from signs.h5).
# Run INSIDE the container (from anywhere):
#     bash analyze/paper_extra/run_c_train.sh <seed> <gpu>     # e.g. 0 4 / 1 5 / 2 7
# Only GPU4, 5, 7 are allowed (GPU0-3 and 6 are never used for C).
set -u
cd "$(dirname "$0")/../.."
seed=$1 gpu=$2
case $gpu in 4|5|7) ;; *) echo "GPU $gpu is not allowed for C"; exit 1 ;; esac
test -f data/data/paper_s1_mixed80_signrand/signs.h5 || { echo "signs.h5 missing"; exit 1; }
mse=paper_extra/C_signrand/s1_signrand_mse l1=paper_extra/C_signrand/s1_signrand_l1_s${seed}
test -e data/result/${mse}/${seed} && { echo "data/result/${mse}/${seed} exists; not overwriting"; exit 1; }
mkdir -p data/result/${mse} data/result/${l1}
setsid nohup bash -c "
    CUDA_VISIBLE_DEVICES=${gpu} python -u analyze/paper_extra/train_paper_c.py --exp_config ${mse} --seed ${seed} \
        --device cuda:0 --cudnn_deterministic --data_load_memory > data/result/${mse}/train_seed${seed}.log 2>&1 &&
    test -f data/result/${mse}/${seed}/model/00200.pth &&
    CUDA_VISIBLE_DEVICES=${gpu} python -u analyze/paper_extra/train_paper_c.py --exp_config ${l1} --seed ${seed} \
        --device cuda:0 --cudnn_deterministic --data_load_memory > data/result/${l1}/train_seed${seed}.log 2>&1
    echo \"exit \$?\" > data/result/${l1}/chain_seed${seed}.done
" < /dev/null > /dev/null 2>&1 &
echo "launched signrand seed ${seed} on GPU ${gpu}"
