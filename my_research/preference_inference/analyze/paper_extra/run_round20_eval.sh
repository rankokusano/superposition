#!/bin/bash
# Paper round 20 Sec.2 full evaluation (docs/paper_extra_log.md Sec.44.3): worker <w> of <n> (round20_eval.py --worker).
# Run INSIDE the container, detached:  GPU=<g> setsid nohup bash analyze/paper_extra/run_round20_eval.sh <w> <n> > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:?}
[ "$GPU" = "0" ] && { echo "GPU0 is not to be used"; exit 1; }
export OMP_NUM_THREADS=2
L=data/result/paper_extra/Round20/logs; mkdir -p $L
nice -n 10 python -u -W ignore analyze/paper_extra/round20_eval.py --worker $1 --n_workers $2 > $L/eval_w$1.log 2>&1
echo "exit $? tracebacks $(grep -c Traceback $L/eval_w$1.log) $(date '+%m-%d %H:%M')" > $L/done_w$1
