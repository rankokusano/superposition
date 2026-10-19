#!/bin/bash
# v11 stage-2 evaluation of the spare pair s3 (docs/v11_experiment_log.md Sec.7.56, Sec.7.42 / 7.51-7.53 methods),
# GPU1. Run INSIDE the container from the v11 worktree's my_research/preference_inference.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=1
L=data/result/v11/stage2/logs_s3; mkdir -p $L
for st in green pref; do
  python analyze/v11/check_convergence_v11.py --exp_fmt v11_s2_${st}_{c}_s{s} --seeds 3 --sub stage2/convergence_${st}_s3 > $L/conv_${st}.log 2>&1
done
R="green:random:3 green:mixed:3 pref:random:3 pref:mixed:3"
python -u analyze/v11/eval_s2_v11.py --runs $R > $L/eval_s2.log 2>&1; echo "eval_s2 exit $?" >> $L/done
E=data/result/v11/stage2/eval; mkdir -p $E/rep_s3; cp $E/green_mixed_s3_ep200.json $E/green_mixed_s3_ep200_per.npz $E/rep_s3/
python -u analyze/v11/eval_s2_v11.py --runs green:mixed:3 --epochs 200 > $L/eval_s2_rep.log 2>&1; echo "eval_s2_rep exit $?" >> $L/done
python - >> $L/done 2>&1 <<'PY'
import json, numpy as np
E = 'data/result/v11/stage2/eval'
a, b = json.load(open(f'{E}/rep_s3/green_mixed_s3_ep200.json')), json.load(open(f'{E}/green_mixed_s3_ep200.json'))
x, y = np.load(f'{E}/rep_s3/green_mixed_s3_ep200_per.npz'), np.load(f'{E}/green_mixed_s3_ep200_per.npz')
print('bit-identical', a == b and all((x[k] == y[k]).all() for k in x))
PY
python -u analyze/v11/t1t3_s2_v11.py --runs $R > $L/t1t3.log 2>&1; echo "t1t3 exit $?" >> $L/done
python -u analyze/v11/t4t6_s2_v11.py --runs $R > $L/t4t6.log 2>&1; echo "t4t6 exit $?" >> $L/done
python -u analyze/v11/m1d_other_v11.py --runs $R > $L/m1d_other.log 2>&1; echo "m1d_other exit $?" >> $L/done
