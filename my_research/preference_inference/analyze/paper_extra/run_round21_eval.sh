#!/bin/bash
# Paper round 21 (docs/paper_extra_log.md Sec.45.2): one evaluation worker inside Docker.
# Usage: GPU=<1|2|3> setsid nohup bash analyze/paper_extra/run_round21_eval.sh <worker> <n_workers> &
set -u
cd "$(dirname "$0")/../.."
W=$1; NW=$2
[ "${GPU}" = "0" ] && { echo "GPU0 is not used"; exit 1; }
LOG=data/result/paper_extra/Round21/logs
mkdir -p $LOG
docker exec -e CUDA_VISIBLE_DEVICES=${GPU} -e OMP_NUM_THREADS=2 kusano_research bash -c \
  "cd /work/.worktrees/v11/my_research/preference_inference && nice -n 10 python -W ignore analyze/paper_extra/round21_eval.py main --worker $W --n_workers $NW" \
  > $LOG/eval_w${W}.log 2>&1
echo "exit $? tracebacks $(grep -c Traceback $LOG/eval_w${W}.log) $(date +%m-%d\ %H:%M)" > $LOG/done_w${W}
