#!/bin/bash
# Paper round 11, Sec.35.8 D / Sec.35.9: position error for E1's signrand50 (L1 stage, late 5, A-1 Red, 4 sets) and
# ep200 again (_rep), after Sec.35.8's runs and E1's evaluation (E2 FP of sr50 by run_e1_eval.sh). GPU1 (Kusano-san).
# Run INSIDE the container, detached:  GPU=1 setsid nohup bash analyze/paper_extra/run_e2pos_sr50.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:?}
[ "$GPU" = "0" ] && { echo "GPU0 is not to be used"; exit 1; }
L=data/result/paper_extra/E2/pos/logs; mkdir -p $L
R=analyze/paper_extra/e2pos_run.py
for s in 0 1 2 3 4 5; do
    python -u $R --models sr50:$s > $L/pos_sr50_s$s.log 2>&1
    python -u $R --models sr50:$s --epochs 200 --tag _rep > $L/pos_rep_sr50_s$s.log 2>&1
done
echo "tracebacks $(grep -l Traceback $L/pos_sr50_*.log $L/pos_rep_sr50_*.log | wc -l) $(date '+%m-%d %H:%M')" > $L/pos_sr50.done
