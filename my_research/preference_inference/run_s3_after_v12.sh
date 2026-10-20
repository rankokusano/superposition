#!/bin/bash
# v12 spare pair s3 (docs/v12_experiment_log.md Sec.26, 28, 35): after s3's stage 1, the selection criterion; if met,
# s3's stage 2 (Sec.28) on GPU1/2, s3's stage-1 values (A on GPU2, G5 on GPU1; reported separately), the stage-2
# evaluation, then s3's stage 2b and the shape-A reference for s0 on GPU1/2. If not met, only the shape-A reference.
# Run INSIDE the container, detached. GPU0 never used.
set -u
cd "$(dirname "$0")"
L=data/result/v12/logs/s3; mkdir -p $L
until [ -f data/result/v12_s1_random_l1_s3/chain_seed3.done ] && [ -f data/result/v12_s1_mixed_l1_s3/chain_seed3.done ]; do sleep 120; done
echo "s3 chains: $(cat data/result/v12_s1_random_l1_s3/chain_seed3.done) / $(cat data/result/v12_s1_mixed_l1_s3/chain_seed3.done) mse ckpts $(ls data/result/v12_s1_random_mse/3/model | wc -l)/$(ls data/result/v12_s1_mixed_mse/3/model | wc -l) l1 ckpts $(ls data/result/v12_s1_random_l1_s3/3/model | wc -l)/$(ls data/result/v12_s1_mixed_l1_s3/3/model | wc -l) $(date)" >> $L/log
CUDA_VISIBLE_DEVICES=1 python analyze/v12/v11fig.py h2 --runs random:3:200 mixed:3:200 --sub ../v12/main/h2_check_s3 > $L/h2.log 2>&1
ok=$(python - <<'PY'
import json
d = json.load(open('data/result/v12/main/h2_check_s3/h2_check.json'))
v = [d[k]['ridge_xy'] for k in ('random:3:200', 'mixed:3:200')]
print('yes' if all(min(x) >= 0.8 for x in v) else 'no', v)
PY
)
echo "s3 selection: $ok" >> $L/log
shapeA() {
  PREFIX=v12_s2a bash run_stage2_v12.sh 0 1 2 >> $L/log 2>&1
  PREFIX=v12_s2a OUT=stage2a setsid nohup bash run_s2eval_v12.sh 0 2 > /dev/null 2>&1 < /dev/null &
}
case "$ok" in
yes*)
  bash run_stage2_v12.sh 3 1 2 >> $L/log 2>&1
  GPU=2 bash run_eval_s1_v12.sh A 3; echo "A s3 exit $?" >> $L/log
  GPU=1 bash run_eval_s1_v12.sh G5 3; echo "G5 s3 exit $?" >> $L/log
  bash run_s2eval_v12.sh 3 1; echo "stage-2 eval s3 done $(date)" >> $L/log
  python analyze/v12/aggregate_s2_v12.py --seeds 0 1 2 3 --tag s0s1s2s3 > $L/aggregate_stage2_s0s1s2s3.log 2>&1
  PREFIX=v12_s2b bash run_stage2_v12.sh 3 1 2 >> $L/log 2>&1
  PREFIX=v12_s2b OUT=stage2b setsid nohup bash run_s2eval_v12.sh 3 1 > /dev/null 2>&1 < /dev/null &
  shapeA ;;
*)
  shapeA ;;
esac
echo "done $(date)" >> $L/log
