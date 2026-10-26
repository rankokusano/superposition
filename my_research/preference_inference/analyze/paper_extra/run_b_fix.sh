#!/bin/bash
# Paper extra B (log Sec.17): re-run only the steps that failed in run_b_eval.sh / run_b_eval_pcols.sh because of
# two bugs in b_copy_run.py (h2: config looked up by the copy-model name; inverse: analyze/v14 not on sys.path).
# Usage INSIDE the container: GPU=6 bash analyze/paper_extra/run_b_fix.sh <seed> <cp|pcols>
set -u
cd "$(dirname "$0")/../.."
GPU=${GPU:-6}; s=$1; mode=$2
export CUDA_VISIBLE_DEVICES=$GPU
B="python -u analyze/paper_extra/b_copy_run.py"
EPS="160 170 180 190 200"
if [ "$mode" = pcols ]; then export B_PREFIX=v13nscpp B_OUT=B_copy_pcols; o=B_copy_pcols; f=v13nscpp
else o=B_copy; f=v13nscp; fi
L=data/result/paper_extra/$o/logs
$B h2 --runs random:$s:200 mixed:$s:200 --sub ../paper_extra/$o/h2_check_s$s > $L/fix_h2_s$s.log 2>&1
$B inverse --models $f:mixed:$s $f:random:$s --epochs $EPS --sets pref --out ../paper_extra/$o/inverse > $L/fix_inverse_s$s.log 2>&1
$B inverse --models ${f}mse:mixed:$s ${f}mse:random:$s --epochs $EPS --sets pref --out ../paper_extra/$o/inverse > $L/fix_inverse_mse_s$s.log 2>&1
echo "tracebacks $(grep -l Traceback $L/fix_*_s$s.log | wc -l)" > $L/fix_seed$s.done
