#!/bin/bash
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:5

gen() {  # exp epoch dataset
  echo "[fig4c] $(date -u) test $1 ep$2 on $3 (state only)"
  python test.py --exp_config "$1" --seed 0 --device cuda:5 \
    --test_epoch "$2" --test_data_name "$3" --test_batch_size 10 --test_modes eval \
    --save_targets state self_position other_position \
    --test_name "${3}_fig4c" --eval_seed 0 --cudnn_deterministic > "logs/fig4c_${1}.log" 2>&1
  echo "[fig4c] $1 rc=$? -> data/result/$1/0/test/${3}_fig4c/save/saved.h5"
  ls -la data/result/$1/0/test/${3}_fig4c/save/saved.h5 2>/dev/null
}

gen exp1_l1_1000 200 self_random_other_stay_1000
gen v5_base_l1   200 r3_stay
gen v5_base_l1   200 r2_a1random_a2rl
gen r3_a_direct_1000pretrain_400 400 r2_a1random_a2rl
echo "[fig4c] DONE $(date -u)"
