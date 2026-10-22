#!/bin/bash
# v14-A inverse inference (docs/v14_experiment_log.md Sec.3, pre-registered). Evaluation only, no training.
# Run INSIDE the container from the v11 worktree's my_research/preference_inference:
#     bash run_inverse_v14.sh <gpu>     (GPU5 or GPU6 only)
# 1. v12 pairs s0..s3 and v11 pairs s0, s2: late-5, sets pref + green -> data/result/v14/inverse/
# 2. ep200 of every model again (tag _rep) for the bit-identical check
# 3. aggregation: main (v12 s0-s2), 4 pairs (with s3), v11 reference
set -u
cd "$(dirname "$0")"
gpu=$1
case $gpu in 5|6) ;; *) echo "GPU $gpu is not allowed for v14"; exit 1 ;; esac
L=data/result/v14/logs
mkdir -p $L data/result/v14/inverse
setsid nohup bash -c "
    export CUDA_VISIBLE_DEVICES=${gpu}
    M='v12:mixed:0 v12:random:0 v12:mixed:1 v12:random:1 v12:mixed:2 v12:random:2 v12:mixed:3 v12:random:3 v11:mixed:0 v11:random:0 v11:mixed:2 v11:random:2'
    python -u analyze/v14/eval_inverse_v14.py --models \$M --out inverse > $L/inverse_eval.log 2>&1 &&
    python -u analyze/v14/eval_inverse_v14.py --models \$M --epochs 200 --tag_suffix _rep --out inverse > $L/inverse_rep.log 2>&1 &&
    python -u analyze/v14/aggregate_inverse_v14.py --pairs v12:mixed:0,v12:random:0 v12:mixed:1,v12:random:1 v12:mixed:2,v12:random:2 \
        --name aggregate_v12_s0s1s2 > $L/inverse_agg.log 2>&1 &&
    python -u analyze/v14/aggregate_inverse_v14.py --pairs v12:mixed:0,v12:random:0 v12:mixed:1,v12:random:1 v12:mixed:2,v12:random:2 \
        v12:mixed:3,v12:random:3 --name aggregate_v12_s0s1s2s3 >> $L/inverse_agg.log 2>&1 &&
    python -u analyze/v14/aggregate_inverse_v14.py --pairs v11:mixed:0,v11:random:0 v11:mixed:2,v11:random:2 \
        --name aggregate_v11_s0s2 >> $L/inverse_agg.log 2>&1
    echo \"exit \$?\" > $L/inverse.done
" < /dev/null > /dev/null 2>&1 &
echo "launched v14-A on GPU ${gpu}"
