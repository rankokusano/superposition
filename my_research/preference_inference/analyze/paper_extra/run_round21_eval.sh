#!/bin/bash
# Paper round 21 Sec.2 evaluation (docs/paper_extra_log.md Sec.45.2): worker <w> of <n> (round21_eval.py main).
# Run INSIDE the container, detached:  GPU=<g> setsid nohup bash analyze/paper_extra/run_round21_eval.sh <w> <n> > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:?}
[ "$GPU" = "0" ] && { echo "GPU0 is not to be used"; exit 1; }
export OMP_NUM_THREADS=2
L=data/result/paper_extra/Round21/logs; mkdir -p $L
nice -n 10 python -u -W ignore analyze/paper_extra/round21_eval.py main --worker $1 --n_workers $2 > $L/eval_w$1.log 2>&1
echo "exit $? tracebacks $(grep -c Traceback $L/eval_w$1.log) $(date '+%m-%d %H:%M')" > $L/done_w$1
