#!/bin/bash
# v13 spare pair s3 (docs/v13_experiment_log.md Sec.3.5, 22): stage-1 evaluation on GPU1, then the selection
# criterion (h2 -> A-2 position, x and y >= 0.8 for both conditions); if met, the pair's experienced stage-2
# runs (both on GPU1; no-experience stage 2 = v12 s3, reused) and their evaluation. Run INSIDE the container.
set -u
cd "$(dirname "$0")"
L=data/result/v13/logs/eval_s1; mkdir -p $L
GPU=1 bash run_eval_s1_v13.sh PAIR 3
ok=$(python - <<'PY'
import json, glob
f = sorted(glob.glob('data/result/v13/main/h2_check_s3/*.json'))[-1]
d = json.load(open(f))
vals = [d[k]['ridge_xy'] for k in ('random:3:200', 'mixed:3:200')]
print('yes' if all(min(v) >= 0.8 for v in vals) else 'no', vals)
PY
)
echo "s3 selection: $ok $(date)" >> $L/s3_after.log
case "$ok" in yes*)
  bash run_stage2_v13.sh 3 1 1 >> $L/s3_after.log 2>&1
  bash run_s2eval_v13.sh 3 1 ;;
esac
echo "s3 after done $(date)" >> $L/s3_after.log
