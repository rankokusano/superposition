#!/bin/bash
# v12 E1/E2 (docs/v12_experiment_log.md Sec.53, post-hoc check for the discussion): mse 200 ep -> l1 200 ep, seed 0.
# E1 on GPU3, E2 on GPU4 (never GPU0). Run INSIDE the container, detached.
set -u
cd "$(dirname "$0")"
run() { local gpu=$1 base=$2
  for st in mse l1; do exp=${base}_${st}; mkdir -p data/result/$exp
    CUDA_VISIBLE_DEVICES=$gpu python -u train_v12.py --exp_config $exp --seed 0 --device cuda:0 --cudnn_deterministic \
      --data_load_memory > data/result/$exp/train_seed0.log 2>&1
    echo "exit $?" > data/result/$exp/done_seed0
  done; }
run 3 v12_e1_paper_random &
run 4 v12_e2_full_mixed &
wait
