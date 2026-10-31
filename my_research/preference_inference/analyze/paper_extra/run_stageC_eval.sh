#!/bin/bash
# Paper round 14 stage C full evaluation (docs/paper_extra_log.md Sec.38.5, 38.9): stageC_eval.py --models <specs> on one GPU.
# Run INSIDE the container, detached:  GPU=<g> setsid nohup bash analyze/paper_extra/run_stageC_eval.sh <tag> <spec>... > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:?}
[ "$GPU" = "0" ] && { echo "GPU0 is not to be used"; exit 1; }
tag=$1; shift
L=data/result/paper_extra/StageC/logs; mkdir -p $L
python -u analyze/paper_extra/stageC_eval.py --models "$@" > $L/eval_$tag.log 2>&1
echo "exit $? tracebacks $(grep -c Traceback $L/eval_$tag.log) $(date '+%m-%d %H:%M')" > $L/done_$tag
