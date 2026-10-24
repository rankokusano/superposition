#!/bin/bash
# v13 addition (docs/v13_experiment_log.md Sec.31): one stage-1 run (mse 200ep -> l1 200ep) of an extra pair,
# saved under the existing run names in a new seed directory. Never writes into an existing seed directory.
# Usage (INSIDE the container from my_research/preference_inference): bash run_extra_v13.sh <run> <seed> <gpu>
#   run: v12random | v12mixed | v13mixed      GPUs allowed: 1 2 3 5 6 7 (GPU0 and GPU4 are not ours)
set -u
cd "$(dirname "$0")"
run=$1; seed=$2; gpu=$3
case $gpu in 1|2|3|5|6|7) ;; *) echo "GPU $gpu is not allowed"; exit 1 ;; esac
case $run in
  v12random) mse=v12_s1_random_mse; l1=v12_s1_random_l1_s${seed} ;;
  v12mixed)  mse=v12_s1_mixed_mse;  l1=v12_s1_mixed_l1_s${seed} ;;
  v13mixed)  mse=v13_s1_mixed_mse;  l1=v13_s1_mixed_l1_s${seed} ;;
  *) echo "unknown run $run"; exit 1 ;;
esac
for d in data/result/${mse}/${seed} data/result/${l1}/${seed}; do
  test -e $d && { echo "$d exists; not overwriting"; exit 1; }
done
test -f config/exp/${l1}.yml || { echo "missing config ${l1}"; exit 1; }
mkdir -p data/result/${l1}
setsid nohup bash -c "
    CUDA_VISIBLE_DEVICES=${gpu} python -u train_v12.py --exp_config ${mse} --seed ${seed} --device cuda:0 \
        --cudnn_deterministic --data_load_memory > data/result/${mse}/train_seed${seed}.log 2>&1 &&
    test -f data/result/${mse}/${seed}/model/00200.pth &&
    CUDA_VISIBLE_DEVICES=${gpu} python -u train_v12.py --exp_config ${l1} --seed ${seed} --device cuda:0 \
        --cudnn_deterministic --data_load_memory > data/result/${l1}/train_seed${seed}.log 2>&1
    echo \"exit \$?\" > data/result/${l1}/chain_seed${seed}.done
" < /dev/null > /dev/null 2>&1 &
echo "launched ${mse} -> ${l1} seed ${seed} on GPU ${gpu} at $(date)"
