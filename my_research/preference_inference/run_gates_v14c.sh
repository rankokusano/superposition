#!/bin/bash
# v14-C gate evaluation (docs/v14_experiment_log.md Sec.31). Per pair: s1 (mse late-5) and g5 (l1 late-5) for the
# given conditions, then ep200 again (bit-identical check). Run INSIDE the container:
#     ALLOW_GPU=<gpu> bash run_gates_v14c.sh <gpu> "<seed>:<conds> <seed>:<conds> ..."     e.g. "0:exp,v12noexp,noexpv 2:exp,v12noexp"
# End marker data/result/v14/logs/gates_gpu<gpu>.done.
set -u
cd "$(dirname "$0")"
gpu=$1 jobs=$2
case $gpu in 5|6) ;; *) [ "${ALLOW_GPU:-}" = "$gpu" ] || { echo "GPU $gpu is not allowed for v14"; exit 1; } ;; esac
L=data/result/v14/logs
setsid nohup bash -c "
    export CUDA_VISIBLE_DEVICES=${gpu}
    rc=0
    for j in ${jobs}; do
        s=\${j%%:*}; conds=\$(echo \${j#*:} | tr , ' ')
        for mode in s1 g5; do
            python -u analyze/v14/gates_v14c.py \$mode --seed \$s --conds \$conds > $L/gates_\${mode}_s\${s}.log 2>&1 || rc=1
            python -u analyze/v14/gates_v14c.py \$mode --seed \$s --conds \$conds --epochs 200 --tag _rep > $L/gates_\${mode}_s\${s}_rep.log 2>&1 || rc=1
        done
    done
    echo \"exit \$rc\" > $L/gates_gpu${gpu}.done
" < /dev/null > /dev/null 2>&1 &
echo "v14-C gates on GPU ${gpu}: ${jobs}"
