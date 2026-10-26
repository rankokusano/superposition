#!/bin/bash
# Paper extra C training (docs/paper_extra_log.md Sec.4, 14). NOT launched before Kusano-san confirms.
# Per seed one chain on one GPU: mse 200ep -> l1 200ep (Decoder only, pretrain = same seed's mse ep200),
# signed-pleasure model on v13_s1_mixed80 with
#   signrand : per-episode random sign (signs.h5; Sec.4)      -> data/result/paper_extra/C_signrand/
#   pzero    : pleasure input always 0 (sign 0; Sec.14)       -> data/result/paper_extra/C_pzero/
# Run INSIDE the container (from anywhere):
#     bash analyze/paper_extra/run_c_train.sh <seed> <gpu> [signrand|pzero]     # default signrand
# Only GPU1, 2, 4, 5, 6, 7 are accepted (GPU0 and GPU3 are never used); check that the GPU is free first.
set -u
cd "$(dirname "$0")/../.."
seed=$1 gpu=$2 cond=${3:-signrand}
case $gpu in 1|2|4|5|6|7) ;; *) echo "GPU $gpu is not allowed for C"; exit 1 ;; esac
case $cond in
    signrand) test -f data/data/paper_s1_mixed80_signrand/signs.h5 || { echo "signs.h5 missing"; exit 1; }
              d=C_signrand ;;
    pzero) d=C_pzero ;;
    *) echo "unknown condition $cond"; exit 1 ;;
esac
mse=paper_extra/${d}/s1_${cond}_mse l1=paper_extra/${d}/s1_${cond}_l1_s${seed}
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
echo "launched ${cond} seed ${seed} on GPU ${gpu}"
