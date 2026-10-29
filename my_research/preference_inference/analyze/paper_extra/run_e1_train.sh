#!/bin/bash
# Paper round 11 E1 training (docs/paper_extra_log.md Sec.35.4). NOT launched before Kusano-san confirms the GPUs.
# Per seed one chain on one GPU, as run_c_train.sh: mse 200ep -> l1 200ep (Decoder only, pretrain = same seed's mse
# ep200), signed-pleasure model on v11_s1_mixed (experienced 50%) with the 80% signrand condition's per-episode sign
# -> data/result/paper_extra/E1_signrand50/. Run INSIDE the container:  bash analyze/paper_extra/run_e1_train.sh <seed> <gpu>
# GPU0 is never accepted; the allowed set is given by Kusano-san (ALLOWED below, set before launching).
set -u
cd "$(dirname "$0")/../.."
seed=$1 gpu=$2
ALLOWED="${E1_GPUS:-}"
[ "$gpu" != 0 ] && [[ " $ALLOWED " == *" $gpu "* ]] || { echo "GPU $gpu is not in the confirmed set '$ALLOWED'"; exit 1; }
test -f data/data/paper_s1_mixed50_signrand/signs.h5 || { echo "signs.h5 missing"; exit 1; }
mse=paper_extra/E1_signrand50/s1_signrand50_mse l1=paper_extra/E1_signrand50/s1_signrand50_l1_s${seed}
test -e data/result/${mse}/${seed} && { echo "data/result/${mse}/${seed} exists; not overwriting"; exit 1; }
mkdir -p data/result/${mse} data/result/${l1}
setsid nohup bash -c "
    CUDA_VISIBLE_DEVICES=${gpu} python -u analyze/paper_extra/train_paper_e1.py --exp_config ${mse} --seed ${seed} \
        --device cuda:0 --cudnn_deterministic --data_load_memory > data/result/${mse}/train_seed${seed}.log 2>&1 &&
    test -f data/result/${mse}/${seed}/model/00200.pth &&
    CUDA_VISIBLE_DEVICES=${gpu} python -u analyze/paper_extra/train_paper_e1.py --exp_config ${l1} --seed ${seed} \
        --device cuda:0 --cudnn_deterministic --data_load_memory > data/result/${l1}/train_seed${seed}.log 2>&1
    echo \"exit \$?\" > data/result/${l1}/chain_seed${seed}.done
" < /dev/null > /dev/null 2>&1 &
echo "launched E1 signrand50 seed ${seed} on GPU ${gpu}"
