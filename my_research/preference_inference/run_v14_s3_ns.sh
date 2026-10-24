#!/bin/bash
# v14 Sec.35 on one GPU, in order: (A) v14-A on the v13 spare pair s3 (4-pair judgements), (C) the G5a whose-
# preference check, (B) after the v13-NS stage-1 chains finish (read only), v14-A on v13-NS.
# Run INSIDE the container:  ALLOW_GPU=<gpu> bash run_v14_s3_ns.sh <gpu>
set -u
cd "$(dirname "$0")"
gpu=$1
case $gpu in 5|6) ;; *) [ "${ALLOW_GPU:-}" = "$gpu" ] || { echo "GPU $gpu is not allowed for v14"; exit 1; } ;; esac
L=data/result/v14/logs
setsid nohup bash -c "
    export CUDA_VISIBLE_DEVICES=${gpu}
    E='python -u analyze/v14/eval_inverse_v14.py --out inverse'
    # (A) v13 s3
    \$E --models v13:mixed:3 v13mse:mixed:3 v12mse:random:3 > $L/s3_eval.log 2>&1 &&
    \$E --models v13:mixed:3 v13mse:mixed:3 v12mse:random:3 --epochs 200 --tag_suffix _rep > $L/s3_rep.log 2>&1 &&
    (cd analyze/v14 && A='python -u aggregate_inverse_v14.py --sub inverse' &&
     \$A --pairs v13:mixed:0,v12:random:0 v13:mixed:1,v12:random:1 v13:mixed:2,v12:random:2 v13:mixed:3,v12:random:3 --name aggregate_v13_s0s1s2s3 > ../../$L/s3_agg.log 2>&1 &&
     \$A --pairs v13mse:mixed:0,v12mse:random:0 v13mse:mixed:1,v12mse:random:1 v13mse:mixed:2,v12mse:random:2 v13mse:mixed:3,v12mse:random:3 --name aggregate_v13mse_s0s1s2s3 >> ../../$L/s3_agg.log 2>&1 &&
     \$A --pairs v13:mixed:0,v12:mixed:0 v13:mixed:1,v12:mixed:1 v13:mixed:2,v12:mixed:2 v13:mixed:3,v12:mixed:3 --name compare_v13exp_vs_v12exp_s0s1s2s3 >> ../../$L/s3_agg.log 2>&1 &&
     python -u fp_direction_v14.py --tags v13mse_mixed_3 v12mse_random_3 --name fp_direction_v13mse_s3 > ../../$L/s3_dir.log 2>&1 &&
     python -u fp_direction_v14.py --tags v13_mixed_3 v12_random_3 --epochs 200 --name fp_direction_v13l1_s3 >> ../../$L/s3_dir.log 2>&1)
    echo \"exit \$?\" > $L/s3.done
    # (C) G5a whose-preference check
    (cd analyze/v14 && python -u g5a_whose_v14.py > ../../$L/g5a_whose.log 2>&1)
    echo \"exit \$?\" > $L/g5a_whose.done
    # (B) v13-NS, after its 6 stage-1 chains
    for c in mixed random; do for s in 0 1 2; do
        until [ -e data/result/v13ns_s1_\${c}_l1_s\${s}/chain_seed\${s}.done ]; do sleep 120; done
    done; done
    N='v13ns:mixed:0 v13ns:random:0 v13ns:mixed:1 v13ns:random:1 v13ns:mixed:2 v13ns:random:2'
    NM='v13nsmse:mixed:0 v13nsmse:random:0 v13nsmse:mixed:1 v13nsmse:random:1 v13nsmse:mixed:2 v13nsmse:random:2'
    \$E --models \$N \$NM > $L/ns_eval.log 2>&1 &&
    \$E --models \$N \$NM --epochs 200 --tag_suffix _rep > $L/ns_rep.log 2>&1 &&
    (cd analyze/v14 && A='python -u aggregate_inverse_v14.py --sub inverse' &&
     \$A --pairs v13:mixed:0,v13ns:mixed:0 v13:mixed:1,v13ns:mixed:1 v13:mixed:2,v13ns:mixed:2 --name aggregate_v13_vs_ns > ../../$L/ns_agg.log 2>&1 &&
     \$A --pairs v13ns:mixed:0,v13ns:random:0 v13ns:mixed:1,v13ns:random:1 v13ns:mixed:2,v13ns:random:2 --name aggregate_ns_exp_vs_noexp >> ../../$L/ns_agg.log 2>&1 &&
     \$A --pairs v13mse:mixed:0,v13nsmse:mixed:0 v13mse:mixed:1,v13nsmse:mixed:1 v13mse:mixed:2,v13nsmse:mixed:2 --name aggregate_v13mse_vs_nsmse >> ../../$L/ns_agg.log 2>&1 &&
     \$A --pairs v13nsmse:mixed:0,v13nsmse:random:0 v13nsmse:mixed:1,v13nsmse:random:1 v13nsmse:mixed:2,v13nsmse:random:2 --name aggregate_nsmse_exp_vs_noexp >> ../../$L/ns_agg.log 2>&1 &&
     python -u fp_check_v14.py --pairs v13ns_mixed_0,v13ns_random_0 v13ns_mixed_1,v13ns_random_1 v13ns_mixed_2,v13ns_random_2 --name fp_check_ns > ../../$L/ns_fpcheck.log 2>&1 &&
     python -u fp_direction_v14.py --tags v13nsmse_mixed_0 v13nsmse_random_0 v13nsmse_mixed_1 v13nsmse_random_1 v13nsmse_mixed_2 v13nsmse_random_2 --name fp_direction_nsmse > ../../$L/ns_dir.log 2>&1 &&
     python -u fp_direction_v14.py --tags v13ns_mixed_0 v13ns_random_0 v13ns_mixed_1 v13ns_random_1 v13ns_mixed_2 v13ns_random_2 --name fp_direction_nsl1 >> ../../$L/ns_dir.log 2>&1)
    echo \"exit \$?\" > $L/ns.done
" < /dev/null > /dev/null 2>&1 &
echo "v14 s3 / G5a check / NS chain on GPU ${gpu}"
