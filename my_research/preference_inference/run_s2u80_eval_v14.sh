#!/bin/bash
# v14-B on v13 evaluation chain (docs/v14_experiment_log.md Sec.30). Waits for the 3 runs v14_s2u_pref_mixed80_s{0,1,2},
# then on <gpu>: u / sign agreement (late-5 + ep200 again), judgement against the v12 no-experience v14-B runs
# (aggregate_u_v13), T2/T5 (eval_s2) and T3 (t1t3) via s2run_v14, u vs direction (u_direction_v13) and the extra
# conditions (a) T5 CI, (b) preference over direction (judge_b_extra_v13). Run INSIDE the container:
#     ALLOW_GPU=<gpu> bash run_s2u80_eval_v14.sh <gpu>
set -u
cd "$(dirname "$0")"
gpu=$1
case $gpu in 5|6) ;; *) [ "${ALLOW_GPU:-}" = "$gpu" ] || { echo "GPU $gpu is not allowed for v14"; exit 1; } ;; esac
L=data/result/v14/logs
setsid nohup bash -c "
    for s in 0 1 2; do until [ -e data/result/v14_s2u_pref_mixed80_s\${s}/done_seed\${s} ]; do sleep 60; done; done
    export CUDA_VISIBLE_DEVICES=${gpu}
    R='mixed80:0 mixed80:1 mixed80:2'
    python -u analyze/v14/eval_u_v14.py --runs \$R > $L/s2u80_u.log 2>&1 &&
    python -u analyze/v14/eval_u_v14.py --runs \$R --epochs 200 --tag_suffix _rep > $L/s2u80_u_rep.log 2>&1 &&
    (cd analyze/v14 && python -u aggregate_u_v14.py --seeds 0 1 2 --exp_cond mixed80 --name aggregate_u_v13 > ../../$L/s2u80_agg.log 2>&1) &&
    python -u analyze/v14/s2run_v14.py eval_s2 --s2prefix v14_s2u --out ../v14/stage2u \
        --runs pref:mixed80:0 pref:mixed80:1 pref:mixed80:2 > $L/s2u80_eval_s2.log 2>&1 &&
    python -u analyze/v14/s2run_v14.py t1t3 --s2prefix v14_s2u --out ../v14/stage2u \
        --runs pref:mixed80:0 pref:mixed80:1 pref:mixed80:2 > $L/s2u80_t1t3.log 2>&1 &&
    (cd analyze/v14 && python -u u_direction_v14.py --conds mixed80 random --name u_direction_v13 > ../../$L/s2u80_dir.log 2>&1 &&
        python -u judge_b_extra_v14.py --exp_cond mixed80 --dir_name u_direction_v13 --name judge_b_extra_v13 > ../../$L/s2u80_extra.log 2>&1)
    echo \"exit \$?\" > $L/s2u80_eval.done
" < /dev/null > /dev/null 2>&1 &
echo "v14-B (v13) evaluation chain waiting; GPU ${gpu}"
