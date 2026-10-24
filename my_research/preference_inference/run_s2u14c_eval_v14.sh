#!/bin/bash
# v14-B on v14-C evaluation chain (docs/v14_experiment_log.md Sec.36). Waits for the 3 runs v14_s2u_pref_mixedpref_s{0,1,2},
# then on <gpu>: u / sign agreement (late-5 + ep200 again), judgement against the v12 no-experience v14-B runs
# (aggregate_u_v14c_vs_v12noexp), T2/T5 (eval_s2) and T3 (t1t3) via s2run_v14, u vs direction (u_direction_v14c) and the extra
# conditions (a) T5 CI, (b) preference over direction (judge_b_extra_v14c). Run INSIDE the container:
#     ALLOW_GPU=<gpu> bash run_s2u14c_eval_v14.sh <gpu>
set -u
cd "$(dirname "$0")"
gpu=$1
case $gpu in 5|6) ;; *) [ "${ALLOW_GPU:-}" = "$gpu" ] || { echo "GPU $gpu is not allowed for v14"; exit 1; } ;; esac
L=data/result/v14/logs
setsid nohup bash -c "
    for s in 0 1 2; do until [ -e data/result/v14_s2u_pref_mixedpref_s\${s}/done_seed\${s} ]; do sleep 60; done; done
    export CUDA_VISIBLE_DEVICES=${gpu}
    R='mixedpref:0 mixedpref:1 mixedpref:2'
    python -u analyze/v14/eval_u_v14.py --runs \$R > $L/s2u14c_u.log 2>&1 &&
    python -u analyze/v14/eval_u_v14.py --runs \$R --epochs 200 --tag_suffix _rep > $L/s2u14c_u_rep.log 2>&1 &&
    (cd analyze/v14 && python -u aggregate_u_v14.py --seeds 0 1 2 --exp_cond mixedpref --name aggregate_u_v14c_vs_v12noexp > ../../$L/s2u14c_agg.log 2>&1 &&
        python -u aggregate_u_v14.py --seeds 0 1 2 --exp_cond mixedpref --partner_cond mixed --name aggregate_u_v14c_vs_v12exp >> ../../$L/s2u14c_agg.log 2>&1) &&
    python -u analyze/v14/s2run_v14.py eval_s2 --s2prefix v14_s2u --out ../v14/stage2u \
        --runs pref:mixedpref:0 pref:mixedpref:1 pref:mixedpref:2 > $L/s2u14c_eval_s2.log 2>&1 &&
    python -u analyze/v14/s2run_v14.py t1t3 --s2prefix v14_s2u --out ../v14/stage2u \
        --runs pref:mixedpref:0 pref:mixedpref:1 pref:mixedpref:2 > $L/s2u14c_t1t3.log 2>&1 &&
    (cd analyze/v14 && python -u u_direction_v14.py --conds mixedpref random mixed --name u_direction_v14c > ../../$L/s2u14c_dir.log 2>&1 &&
        python -u judge_b_extra_v14.py --exp_cond mixedpref --dir_name u_direction_v14c --name judge_b_extra_v14c > ../../$L/s2u14c_extra.log 2>&1)
    echo \"exit \$?\" > $L/s2u14c_eval.done
" < /dev/null > /dev/null 2>&1 &
echo "v14-B (v14-C) evaluation chain waiting; GPU ${gpu}"
