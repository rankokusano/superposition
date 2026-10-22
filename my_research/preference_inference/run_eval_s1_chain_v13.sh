#!/bin/bash
# v13 stage-1: per pair, wait for the experienced chain (mse -> l1), check the expected outputs, then run
# the pair's evaluation on the pair's GPU (s0 -> 7, s1 -> 1, s2 -> 2); the spare pair s3 (Sec.3.5) is
# launched on GPU1 when the three main chains are done. Finally the aggregation (GPU7).
# Run INSIDE the container (setsid nohup) from my_research/preference_inference.
set -u
cd "$(dirname "$0")"
L=data/result/v13/logs/eval_s1; mkdir -p $L
declare -A G=( [0]=7 [1]=1 [2]=2 )
for s in 0 1 2; do
  (
    until [ -f data/result/v13_s1_mixed_l1_s$s/chain_seed$s.done ]; do sleep 120; done
    echo "s$s: chain $(cat data/result/v13_s1_mixed_l1_s$s/chain_seed$s.done) mse ckpts $(ls data/result/v13_s1_mixed_mse/$s/model | wc -l) l1 ckpts $(ls data/result/v13_s1_mixed_l1_s$s/$s/model | wc -l) tracebacks $(cat data/result/v13_s1_mixed_mse/train_seed$s.log data/result/v13_s1_mixed_l1_s$s/train_seed$s.log | grep -ac Traceback) at $(date)" >> $L/completion_check.log
    GPU=${G[$s]} bash run_eval_s1_v13.sh PAIR $s
  ) &
done
until [ $(ls data/result/v13_s1_mixed_l1_s{0,1,2}/chain_seed*.done 2>/dev/null | wc -l) -ge 3 ]; do sleep 120; done
bash run_main_v13.sh 3 1 >> $L/completion_check.log 2>&1
wait
GPU=7 bash run_eval_s1_v13.sh AGG
echo "chain done $(date)" > $L/chain.done
