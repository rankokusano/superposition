#!/bin/bash
# v15 stage 2 trial, no-experience chains only (Sec.20.4, plan C' data): args GPU for noexp_v4, GPU for noexp_v15i.
# Same chain as run_s2_v15.sh: phase 1 (mse 150 ep) -> phase 2 (G only, l1 150 ep), seed 0. Run INSIDE the container, detached.
set -u
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs; mkdir -p $L
chain() { local tag=$1 gpu=$2
  for exp in v15_s2p1_${tag}_mse v15_s2p2_${tag}; do mkdir -p data/result/$exp
    CUDA_VISIBLE_DEVICES=$gpu python -u train_v15.py --exp_config $exp --seed 0 --device cuda:0 --cudnn_deterministic \
      --data_load_memory > data/result/$exp/train_seed0.log 2>&1
    rc=$?; echo "exit $rc" > data/result/$exp/done_seed0
    echo "$exp train exit $rc $(date)" >> $L/done
    [ $rc = 0 ] || return
  done; }
chain noexp_v4 $1 &
chain noexp_v15i $2 &
wait
echo NOEXP_DONE >> $L/done
