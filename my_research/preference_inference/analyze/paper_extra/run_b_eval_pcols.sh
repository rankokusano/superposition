#!/bin/bash
# Paper extra B, pleasure-columns-only copy (docs/paper_extra_log.md Sec.13): run_b_eval.sh with v13nscpp models. NOT launched before Kusano-san confirms.
# Run INSIDE the container:  GPU=6 bash analyze/paper_extra/run_b_eval_pcols.sh <seed>   (one seed; no repro step:
#                            the original NS / 80% / no-experience values are those of Sec.3.4)
# The same steps as run_eval_ns_v13.sh, on the copy models (analyze/paper_extra/b_copy_run.py).
set -u
cd "$(dirname "$0")/../.."
GPU=${GPU:-6}
case $GPU in 4|5|6|7) ;; *) echo "GPU $GPU is not allowed"; exit 1 ;; esac
export CUDA_VISIBLE_DEVICES=$GPU
export B_PREFIX=v13nscpp B_OUT=B_copy_pcols
L=data/result/paper_extra/B_copy_pcols/logs; mkdir -p $L
V="python -u analyze/v13/v12run.py"
B="python -u analyze/paper_extra/b_copy_run.py"
EPS="160 170 180 190 200"
s=$1
$B trial --exp_fmt v13nscpp_s1_{}_mse --seed $s --epochs $EPS --no_p2 --out ../paper_extra/B_copy_pcols/s1eval/s$s > $L/trial_s$s.log 2>&1
$B trial --exp_fmt v13nscpp_s1_{}_mse --seed $s --epochs 200 --no_p2 --tag _rep --out ../paper_extra/B_copy_pcols/s1eval/s$s > $L/trial_rep_s$s.log 2>&1
$B h2 --runs random:$s:200 mixed:$s:200 --sub ../paper_extra/B_copy_pcols/h2_check_s$s > $L/h2_s$s.log 2>&1
for e in $EPS; do
    $B g5 --exp_fmt v13nscpp_s1_{}_l1_s$s --seed $s --epoch $e --sub ../paper_extra/B_copy_pcols/g5/s$s > $L/g5_s${s}_e$e.log 2>&1
done
$B g5 --exp_fmt v13nscpp_s1_{}_l1_s$s --seed $s --epoch 200 --sub ../paper_extra/B_copy_pcols/g5_rep/s$s > $L/g5_rep_s$s.log 2>&1
$B d6 --models nscpp_mixed:$s nscpp_random:$s --sets green pref --out s$s > $L/d6_s$s.log 2>&1
$B decomp --seed $s > $L/decomp_s$s.log 2>&1
$B inverse --models v13nscpp:mixed:$s v13nscpp:random:$s --epochs $EPS --sets pref --out ../paper_extra/B_copy_pcols/inverse > $L/inverse_s$s.log 2>&1
$B inverse --models v13nscppmse:mixed:$s v13nscppmse:random:$s --epochs $EPS --sets pref --out ../paper_extra/B_copy_pcols/inverse > $L/inverse_mse_s$s.log 2>&1
echo "tracebacks $(grep -l Traceback $L/*_s$s*.log | wc -l)" > $L/seed$s.done
