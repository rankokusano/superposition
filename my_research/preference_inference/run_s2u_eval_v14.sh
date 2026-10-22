#!/bin/bash
# v14-B evaluation chain (docs/v14_experiment_log.md Sec.4, 10, 15). Waits for the 6 one-bit runs (done markers),
# then on <gpu>: u / sign agreement (late-5 on pref + green test, and ep200 again for the bit-identical check),
# the v12 stage-2 evaluation (T2, T5, correlations) and T1/T3 via s2run_v14 (outputs data/result/v14/stage2u/),
# and the judgement (aggregate_u_v14). Run INSIDE the container:  bash run_s2u_eval_v14.sh <gpu>
set -u
cd "$(dirname "$0")"
gpu=$1
case $gpu in 5|6) ;; *) echo "GPU $gpu is not allowed for v14"; exit 1 ;; esac
L=data/result/v14/logs
mkdir -p $L
setsid nohup bash -c "
    for c in mixed random; do for s in 0 1 2; do
        until [ -e data/result/v14_s2u_pref_\${c}_s\${s}/done_seed\${s} ]; do sleep 60; done
    done; done
    export CUDA_VISIBLE_DEVICES=${gpu}
    R='mixed:0 random:0 mixed:1 random:1 mixed:2 random:2'
    python -u analyze/v14/eval_u_v14.py --runs \$R > $L/s2u_u.log 2>&1 &&
    python -u analyze/v14/eval_u_v14.py --runs \$R --epochs 200 --tag_suffix _rep > $L/s2u_u_rep.log 2>&1 &&
    python -u analyze/v14/aggregate_u_v14.py --seeds 0 1 2 --name aggregate_u_s0s1s2 > $L/s2u_agg.log 2>&1 &&
    python -u analyze/v14/s2run_v14.py eval_s2 --s2prefix v14_s2u --out ../v14/stage2u \
        --runs pref:mixed:0 pref:random:0 pref:mixed:1 pref:random:1 pref:mixed:2 pref:random:2 d6:data:0 > $L/s2u_eval_s2.log 2>&1 &&
    python -u analyze/v14/s2run_v14.py t1t3 --s2prefix v14_s2u --out ../v14/stage2u \
        --runs pref:mixed:0 pref:random:0 pref:mixed:1 pref:random:1 pref:mixed:2 pref:random:2 > $L/s2u_t1t3.log 2>&1
    echo \"exit \$?\" > $L/s2u_eval.done
" < /dev/null > /dev/null 2>&1 &
echo "v14-B evaluation chain waiting; GPU ${gpu}"
