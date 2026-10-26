#!/bin/bash
# v15 2(b) (Sec.24.5): phase 2 re-trained with A-2 hidden in the input view. Args: GPU for the v4 pair, GPU for the v15i pair
# (two runs per card), then the 2(b) evaluation of each run on the same GPU. Run INSIDE the container, detached.
set -u
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs; mkdir -p $L
one() { local tag=$1 gpu=$2 exp=v15_s2p2h_$1
  mkdir -p data/result/$exp
  CUDA_VISIBLE_DEVICES=$gpu python -u train_v15.py --exp_config $exp --seed 0 --device cuda:0 --cudnn_deterministic \
    --data_load_memory > data/result/$exp/train_seed0.log 2>&1
  rc=$?; echo "exit $rc" > data/result/$exp/done_seed0; echo "$exp train exit $rc $(date)" >> $L/done_2b
  [ $rc = 0 ] || return
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s2_hide_v15.py --exp $exp --device cuda:0 \
    > $L/hide_$exp.log 2>&1
  rc=$?; echo "$exp eval exit $rc $(date)" >> $L/done_2b
}
one exp_v4 $1 & one noexp_v4 $1 & one exp_v15i $2 & one noexp_v15i $2 &
wait
echo ALL_DONE_2B >> $L/done_2b
