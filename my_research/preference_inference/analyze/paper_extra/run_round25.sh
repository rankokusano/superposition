#!/bin/bash
# Paper round 25 trial (docs/paper_extra_log.md Sec.49.2): one condition on one GPU, two chains in parallel:
#   chain A: stage 1 M (100 epochs) -> stage 2 LSTM lr x0.1 (50 epochs, from stage-1 M ep100)
#   chain B: stage 1 N (100 epochs) -> wait for stage 1 M -> stage 2 LSTM lr x0.01
# Run INSIDE the container, detached:  GPU=<g> setsid nohup bash analyze/paper_extra/run_round25.sh <cond> > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:?}
[ "$GPU" = "0" ] && { echo "GPU0 is not to be used"; exit 1; }
export OMP_NUM_THREADS=2
C=$1
R=data/result/paper_extra/Round25
L=$R/logs; mkdir -p $L
T="nice -n 10 python -u -W ignore analyze/paper_extra/round25_train.py"
(
  $T stage1 --cond $C --variant M > $L/stage1_${C}_M.log 2>&1
  $T stage2 --cond $C --ratio 0.1 > $L/stage2_${C}_r0.1.log 2>&1
  echo "exit $? $(date '+%m-%d %H:%M')" > $L/done_${C}_A
) &
(
  $T stage1 --cond $C --variant N > $L/stage1_${C}_N.log 2>&1
  until [ -e $R/runs/stage1_${C}_M/done.json ]; do sleep 60; done
  $T stage2 --cond $C --ratio 0.01 > $L/stage2_${C}_r0.01.log 2>&1
  echo "exit $? $(date '+%m-%d %H:%M')" > $L/done_${C}_B
) &
wait
