#!/bin/bash
# v15 stage 2 trial (docs/v15_experiment_log.md Sec.16): four chains, one per GPU (args: GPUs for exp_v4 noexp_v4 exp_v15i noexp_v15i),
# each phase 1 (mse 150 ep) -> phase 2 (G only, l1 150 ep), seed 0. Waits for the trial datasets. Run INSIDE the container, detached.
set -u
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs; mkdir -p $L
until grep -q "D exit 0" logs/v15_s2data_driver.log 2>/dev/null; do sleep 120; done
chain() { local tag=$1 gpu=$2
  for exp in v15_s2p1_${tag}_mse v15_s2p2_${tag}; do mkdir -p data/result/$exp
    CUDA_VISIBLE_DEVICES=$gpu python -u train_v15.py --exp_config $exp --seed 0 --device cuda:0 --cudnn_deterministic \
      --data_load_memory > data/result/$exp/train_seed0.log 2>&1
    rc=$?; echo "exit $rc" > data/result/$exp/done_seed0
    echo "$exp train exit $rc $(date)" >> $L/done
    [ $rc = 0 ] || return
  done; }
chain exp_v4 $1 &
chain noexp_v4 $2 &
chain exp_v15i $3 &
chain noexp_v15i $4 &
wait
echo ALL_DONE >> $L/done
