#!/bin/bash
# Paper round 23 Sec.2 evaluation (docs/paper_extra_log.md Sec.47.2): part r20|dir, worker <w> of <n> (round23_eval.py main).
# Run INSIDE the container, detached:  GPU=<g> setsid nohup bash analyze/paper_extra/run_round23_eval.sh <part> <w> <n> > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:?}
[ "$GPU" = "0" ] && { echo "GPU0 is not to be used"; exit 1; }
export OMP_NUM_THREADS=2
L=data/result/paper_extra/Round23/logs; mkdir -p $L
nice -n 10 python -u -W ignore analyze/paper_extra/round23_eval.py main --part $1 --worker $2 --n_workers $3 > $L/eval_$1_w$2.log 2>&1
echo "exit $? tracebacks $(grep -c Traceback $L/eval_$1_w$2.log) $(date '+%m-%d %H:%M')" > $L/done_$1_w$2
