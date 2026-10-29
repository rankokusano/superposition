#!/bin/bash
# Paper round 11 E2 (docs/paper_extra_log.md Sec.35.2): all models of pairs 1-6, both stages, late 5 points, 4 sets,
# then ep200 again (_rep) for the bit-identity check. One GPU (Kusano-san: GPU1).
# Run INSIDE the container, detached:  GPU=1 setsid nohup bash analyze/paper_extra/run_e2.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:?}
L=data/result/paper_extra/E2/logs; mkdir -p $L
for f in p50 none p80 sr80; do
    for s in 0 1 2 3 4 5; do
        python -u analyze/paper_extra/e2_run.py --models $f:$s > $L/e2_${f}_s$s.log 2>&1
    done
done
for f in p50 none p80 sr80; do
    for s in 0 1 2 3 4 5; do
        python -u analyze/paper_extra/e2_run.py --models $f:$s --epochs 200 --tag _rep > $L/e2_rep_${f}_s$s.log 2>&1
    done
done
echo "tracebacks $(grep -l Traceback $L/e2_*.log | wc -l) $(date '+%m-%d %H:%M')" > $L/e2_all.done
