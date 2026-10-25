#!/bin/bash
# v14 Sec.43 (pre-registered): v14-A on the extra pairs s4, s5 (v13 exp, v12 no-exp, v12 exp; l1 and mse late-5,
# ep200 again), 6-pair robustness listings (s0-s5) and the FP / direction conditions per pair.
# Run INSIDE the container:  bash run_inverse_s45_v14.sh <gpu>   (GPU5/6 only)
set -u
cd "$(dirname "$0")"
gpu=$1
case $gpu in 5|6) ;; *) echo "GPU $gpu is not allowed for v14"; exit 1 ;; esac
L=data/result/v14/logs
setsid nohup bash -c "
    export CUDA_VISIBLE_DEVICES=${gpu}
    E='python -u analyze/v14/eval_inverse_v14.py --out inverse'
    M='v13:mixed:4 v12:random:4 v12:mixed:4 v13:mixed:5 v12:random:5 v12:mixed:5 v13mse:mixed:4 v12mse:random:4 v12mse:mixed:4 v13mse:mixed:5 v12mse:random:5 v12mse:mixed:5'
    \$E --models \$M > $L/s45_eval.log 2>&1 &&
    \$E --models \$M --epochs 200 --tag_suffix _rep > $L/s45_rep.log 2>&1 &&
    cd analyze/v14 && A='python -u aggregate_inverse_v14.py --sub inverse' &&
    P13='v13:mixed:0,v12:random:0 v13:mixed:1,v12:random:1 v13:mixed:2,v12:random:2 v13:mixed:3,v12:random:3 v13:mixed:4,v12:random:4 v13:mixed:5,v12:random:5' &&
    \$A --pairs \$P13 --name aggregate_v13_s0to5 > ../../$L/s45_agg.log 2>&1 &&
    \$A --pairs \$(echo \$P13 | sed 's/v13:/v13mse:/g; s/v12:/v12mse:/g') --name aggregate_v13mse_s0to5 >> ../../$L/s45_agg.log 2>&1 &&
    \$A --pairs v12:mixed:0,v12:random:0 v12:mixed:1,v12:random:1 v12:mixed:2,v12:random:2 v12:mixed:3,v12:random:3 v12:mixed:4,v12:random:4 v12:mixed:5,v12:random:5 --name aggregate_v12_s0to5 >> ../../$L/s45_agg.log 2>&1 &&
    \$A --pairs v13:mixed:0,v12:mixed:0 v13:mixed:1,v12:mixed:1 v13:mixed:2,v12:mixed:2 v13:mixed:3,v12:mixed:3 v13:mixed:4,v12:mixed:4 v13:mixed:5,v12:mixed:5 --name compare_v13exp_vs_v12exp_s0to5 >> ../../$L/s45_agg.log 2>&1 &&
    python -u fp_direction_v14.py --tags v13mse_mixed_4 v12mse_random_4 v12mse_mixed_4 v13mse_mixed_5 v12mse_random_5 v12mse_mixed_5 --name fp_direction_mse_s45 > ../../$L/s45_dir.log 2>&1 &&
    python -u fp_direction_v14.py --tags v13_mixed_4 v12_random_4 v12_mixed_4 v13_mixed_5 v12_random_5 v12_mixed_5 --epochs 200 --name fp_direction_l1_s45 >> ../../$L/s45_dir.log 2>&1
    echo \"exit \$?\" > ../../$L/s45.done
" < /dev/null > /dev/null 2>&1 &
echo "v14-A s4/s5 chain on GPU ${gpu}"
