#!/bin/bash
# Paper extra B evaluation (docs/paper_extra_log.md Sec.3). NOT launched before Kusano-san confirms.
# Run INSIDE the container:  GPU=6 bash analyze/paper_extra/run_b_eval.sh repro      (pair 1 ep200 re-measure)
#                            GPU=6 bash analyze/paper_extra/run_b_eval.sh <seed>     (copy models, one seed)
# The same steps as run_eval_ns_v13.sh, on the copy models (analyze/paper_extra/b_copy_run.py).
set -u
cd "$(dirname "$0")/../.."
GPU=${GPU:-6}
case $GPU in 4|5|6|7) ;; *) echo "GPU $GPU is not allowed"; exit 1 ;; esac
export CUDA_VISIBLE_DEVICES=$GPU
L=data/result/paper_extra/B_copy/logs; mkdir -p $L
V="python -u analyze/v13/v12run.py"
B="python -u analyze/paper_extra/b_copy_run.py"
EPS="160 170 180 190 200"
if [ "$1" = repro ]; then   # Sec.3.4: re-measure (i) NS, (iii) 80%, (iv) no experience at pair 1 ep200
    $V g5 --exp_fmt v13ns_s1_{}_l1_s0 --seed 0 --epoch 200 --sub ../paper_extra/B_copy/repro/ns_g5 > $L/repro_ns_g5.log 2>&1
    $V g5 --exp_fmt v13_s1_{}_l1_s0 --seed 0 --epoch 200 --sub ../paper_extra/B_copy/repro/v13_g5 > $L/repro_v13_g5.log 2>&1
    V13_NS=1 $V h2 --runs random:0:200 mixed:0:200 --sub ../paper_extra/B_copy/repro/ns_h2 > $L/repro_ns_h2.log 2>&1
    $V h2 --runs random:0:200 mixed:0:200 --sub ../paper_extra/B_copy/repro/v13_h2 > $L/repro_v13_h2.log 2>&1
    echo "exit $?" > $L/repro.done
    exit 0
fi
s=$1
$B trial --exp_fmt v13nscp_s1_{}_mse --seed $s --epochs $EPS --no_p2 --out ../paper_extra/B_copy/s1eval/s$s > $L/trial_s$s.log 2>&1
$B trial --exp_fmt v13nscp_s1_{}_mse --seed $s --epochs 200 --no_p2 --tag _rep --out ../paper_extra/B_copy/s1eval/s$s > $L/trial_rep_s$s.log 2>&1
$B h2 --runs random:$s:200 mixed:$s:200 --sub ../paper_extra/B_copy/h2_check_s$s > $L/h2_s$s.log 2>&1
for e in $EPS; do
    $B g5 --exp_fmt v13nscp_s1_{}_l1_s$s --seed $s --epoch $e --sub ../paper_extra/B_copy/g5/s$s > $L/g5_s${s}_e$e.log 2>&1
done
$B g5 --exp_fmt v13nscp_s1_{}_l1_s$s --seed $s --epoch 200 --sub ../paper_extra/B_copy/g5_rep/s$s > $L/g5_rep_s$s.log 2>&1
$B d6 --models nscp_mixed:$s nscp_random:$s ns_mixed:$s --sets green pref --out s$s > $L/d6_s$s.log 2>&1
$B decomp --seed $s > $L/decomp_s$s.log 2>&1
$B inverse --models v13nscp:mixed:$s v13nscp:random:$s --epochs $EPS --sets pref --out ../paper_extra/B_copy/inverse > $L/inverse_s$s.log 2>&1
$B inverse --models v13nscpmse:mixed:$s v13nscpmse:random:$s --epochs $EPS --sets pref --out ../paper_extra/B_copy/inverse > $L/inverse_mse_s$s.log 2>&1
echo "tracebacks $(grep -l Traceback $L/*_s$s*.log | wc -l)" > $L/seed$s.done
