#!/bin/bash
# v12 stage-1 main: wait for all 6 chains, check the expected outputs, then run the registered evaluation
# (docs/v12_experiment_log.md Sec.17; run_eval_s1_v12.sh) on GPU1-7 (never GPU0), then aggregate.
# Run INSIDE the container from the v11 worktree's my_research/preference_inference.
set -u
cd "$(dirname "$0")"
L=data/result/v12/logs/eval_s1; mkdir -p $L
until [ $(ls data/result/v12_s1_{random,mixed}_l1_s{0,1,2}/chain_seed*.done 2>/dev/null | wc -l) -ge 6 ]; do sleep 120; done
{
for c in random mixed; do for s in 0 1 2; do
  echo "$c s$s: chain $(cat data/result/v12_s1_${c}_l1_s$s/chain_seed$s.done) mse ckpts $(ls data/result/v12_s1_${c}_mse/$s/model | wc -l) l1 ckpts $(ls data/result/v12_s1_${c}_l1_s$s/$s/model | wc -l) tracebacks $(cat data/result/v12_s1_${c}_mse/train_seed$s.log data/result/v12_s1_${c}_l1_s$s/train_seed$s.log | grep -c Traceback)"
done; done
} > $L/completion_check.log
GPU=3 bash run_eval_s1_v12.sh A 0 &
GPU=4 bash run_eval_s1_v12.sh A 1 &
GPU=5 bash run_eval_s1_v12.sh A 2 &
GPU=6 bash run_eval_s1_v12.sh G5 0 &
GPU=7 bash run_eval_s1_v12.sh G5 1 &
GPU=1 bash run_eval_s1_v12.sh G5 2 &
(GPU=2 bash run_eval_s1_v12.sh FIG5; GPU=2 bash run_eval_s1_v12.sh A_rep 0) &
wait
GPU=3 bash run_eval_s1_v12.sh AGG
echo "chain done" > $L/chain.done
