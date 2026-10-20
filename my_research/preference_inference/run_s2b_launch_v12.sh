#!/bin/bash
# v12 (docs/v12_experiment_log.md Sec.35): after pair s1's stage-2 evaluation, write the 3-pair stage-2 judgement,
# then launch stage 2b (signed_pairs) for s0 (GPU3/4), s2 (GPU5/6), s1 (GPU7) and their evaluators.
# Run INSIDE the container (v11 worktree's my_research/preference_inference), detached. GPU0 never used.
set -u
cd "$(dirname "$0")"
L=data/result/v12/logs/stage2b; mkdir -p $L
until grep -q "all done" data/result/v12/stage2/logs/s1/steps.log 2>/dev/null; do sleep 120; done
python analyze/v12/aggregate_s2_v12.py --seeds 0 1 2 --tag s0s1s2 > $L/aggregate_stage2_s0s1s2.log 2>&1
echo "stage-2 3-pair aggregation exit $? $(date)" >> $L/launch.log
PREFIX=v12_s2b bash run_stage2_v12.sh 0 3 4 >> $L/launch.log 2>&1
PREFIX=v12_s2b bash run_stage2_v12.sh 2 5 6 >> $L/launch.log 2>&1
PREFIX=v12_s2b bash run_stage2_v12.sh 1 7 7 >> $L/launch.log 2>&1
for sg in "0 3" "2 5" "1 7"; do
  set -- $sg
  PREFIX=v12_s2b OUT=stage2b setsid nohup bash run_s2eval_v12.sh $1 $2 > /dev/null 2>&1 < /dev/null &
done
echo "stage 2b launched $(date)" >> $L/launch.log
