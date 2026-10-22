#!/bin/bash
# v14 Sec.21 B/C (pre-registered): the FP secondary on the v13 experienced models counted on the mse-stage late-5.
# 1) now: v14-A outputs for v13_s1_mixed_mse and v12_s1_random_mse (s0-s2, ep160-200; ep200 again for the bit check),
#    FP judgement (aggregate_v13mse_s0s1s2.json, FP rows) and the direction comparison (fp_direction_v13mse.json);
# 2) after the l1 chain (run_inverse_v13_v14.sh, inverse_v13.done): the direction comparison on the l1 point
#    (fp_direction_v13l1.json). Run INSIDE the container: bash run_inverse_v13mse_v14.sh <gpu>   (GPU5/6 only)
set -u
cd "$(dirname "$0")"
gpu=$1
case $gpu in 5|6) ;; *) echo "GPU $gpu is not allowed for v14"; exit 1 ;; esac
L=data/result/v14/logs
setsid nohup bash -c "
    export CUDA_VISIBLE_DEVICES=${gpu}
    M='v13mse:mixed:0 v12mse:random:0 v13mse:mixed:1 v12mse:random:1 v13mse:mixed:2 v12mse:random:2'
    python -u analyze/v14/eval_inverse_v14.py --models \$M --out inverse > $L/inverse_v13mse_eval.log 2>&1 &&
    python -u analyze/v14/eval_inverse_v14.py --models \$M --epochs 200 --tag_suffix _rep --out inverse > $L/inverse_v13mse_rep.log 2>&1 &&
    python -u analyze/v14/aggregate_inverse_v14.py --pairs v13mse:mixed:0,v12mse:random:0 v13mse:mixed:1,v12mse:random:1 \
        v13mse:mixed:2,v12mse:random:2 --name aggregate_v13mse_s0s1s2 > $L/inverse_v13mse_agg.log 2>&1 &&
    (cd analyze/v14 && python -u fp_direction_v14.py --tags v13mse_mixed_0 v12mse_random_0 v13mse_mixed_1 v12mse_random_1 \
        v13mse_mixed_2 v12mse_random_2 --name fp_direction_v13mse > ../../$L/inverse_v13mse_dir.log 2>&1)
    echo \"exit \$?\" > $L/inverse_v13mse.done
    until [ -e $L/inverse_v13.done ]; do sleep 120; done
    (cd analyze/v14 && python -u fp_direction_v14.py --tags v13_mixed_0 v12_random_0 v13_mixed_1 v12_random_1 v13_mixed_2 v12_random_2 \
        --epochs 200 --name fp_direction_v13l1 > ../../$L/inverse_v13l1_dir.log 2>&1)
    echo \"exit \$?\" > $L/inverse_v13l1_dir.done
" < /dev/null > /dev/null 2>&1 &
echo "launched v13 mse-stage FP chain on GPU ${gpu}"
