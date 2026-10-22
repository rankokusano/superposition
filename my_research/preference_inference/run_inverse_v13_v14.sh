#!/bin/bash
# v14-A on the v13 experienced (80%) models (docs/v14_experiment_log.md Sec.12, pre-registered before any v13 value).
# Waits for the v13 stage-1 chains (data/result/v13_s1_mixed_l1_s<s>/chain_seed<s>.done; read only, nothing of v13
# is touched), then on <gpu>: late-5 on pref + green, ep200 again (bit-identical), judgement against the v12
# no-experience models of the same seed (already evaluated in data/result/v14/inverse/), the descriptive v13 - v12
# experienced comparison, and the registered FP secondary. Run INSIDE the container: bash run_inverse_v13_v14.sh <gpu>
set -u
cd "$(dirname "$0")"
gpu=$1
case $gpu in 5|6) ;; *) echo "GPU $gpu is not allowed for v14"; exit 1 ;; esac
L=data/result/v14/logs
mkdir -p $L
setsid nohup bash -c "
    for s in 0 1 2; do until [ -e data/result/v13_s1_mixed_l1_s\${s}/chain_seed\${s}.done ]; do sleep 120; done; done
    export CUDA_VISIBLE_DEVICES=${gpu}
    M='v13:mixed:0 v13:mixed:1 v13:mixed:2'
    python -u analyze/v14/eval_inverse_v14.py --models \$M --out inverse > $L/inverse_v13_eval.log 2>&1 &&
    python -u analyze/v14/eval_inverse_v14.py --models \$M --epochs 200 --tag_suffix _rep --out inverse > $L/inverse_v13_rep.log 2>&1 &&
    python -u analyze/v14/aggregate_inverse_v14.py --pairs v13:mixed:0,v12:random:0 v13:mixed:1,v12:random:1 v13:mixed:2,v12:random:2 \
        --name aggregate_v13_s0s1s2 > $L/inverse_v13_agg.log 2>&1 &&
    python -u analyze/v14/aggregate_inverse_v14.py --pairs v13:mixed:0,v12:mixed:0 v13:mixed:1,v12:mixed:1 v13:mixed:2,v12:mixed:2 \
        --name compare_v13exp_vs_v12exp >> $L/inverse_v13_agg.log 2>&1 &&
    cd analyze/v14 && python -u fp_check_v14.py --pairs v13_mixed_0,v12_random_0 v13_mixed_1,v12_random_1 v13_mixed_2,v12_random_2 \
        --name fp_check_v13 > ../../$L/inverse_v13_fp.log 2>&1
    echo \"exit \$?\" > ../../$L/inverse_v13.done
" < /dev/null > /dev/null 2>&1 &
echo "v14-A on v13 waiting; GPU ${gpu}"
