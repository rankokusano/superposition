#!/bin/bash
# Paper round 11 E2 addition (docs/paper_extra_log.md Sec.35.8, outside the pre-registration): A-2 position error and
# red-form bias. One GPU (Kusano-san: GPU1). Order: (1) the 4 conditions, L1 stage, A-1 Red (+ ep200 again, _rep);
# (2) v14-C (v14c s0-2, v14cr s0-1): MSE stage (FP), L1 stage with A-1 Red and Green; (3) the 4 conditions, L1 stage,
# A-1 Green. The E1 signrand50 runs (Sec.35.8 D) are added after E1's evaluation, separately.
# Run INSIDE the container, detached:  GPU=1 setsid nohup bash analyze/paper_extra/run_e2pos.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:?}
L=data/result/paper_extra/E2/pos/logs; mkdir -p $L
R=analyze/paper_extra/e2pos_run.py
for f in p50 none p80 sr80; do
    for s in 0 1 2 3 4 5; do
        python -u $R --models $f:$s > $L/pos_${f}_s$s.log 2>&1
    done
done
for f in p50 none p80 sr80; do
    for s in 0 1 2 3 4 5; do
        python -u $R --models $f:$s --epochs 200 --tag _rep > $L/pos_rep_${f}_s$s.log 2>&1
    done
done
for m in v14c:0 v14c:1 v14c:2 v14cr:0 v14cr:1; do
    python -u $R --models $m --stages mse > $L/pos_mse_${m/:/_s}.log 2>&1
    python -u $R --models $m > $L/pos_${m/:/_s}.log 2>&1
    python -u $R --models $m --a1 green > $L/pos_a1green_${m/:/_s}.log 2>&1
done
echo "step2 $(date '+%m-%d %H:%M')" > $L/pos_step2.done
for f in p50 none p80 sr80; do
    for s in 0 1 2 3 4 5; do
        python -u $R --models $f:$s --a1 green > $L/pos_a1green_${f}_s$s.log 2>&1
    done
done
echo "tracebacks $(grep -l Traceback $L/pos_*.log | wc -l) $(date '+%m-%d %H:%M')" > $L/pos_all.done
