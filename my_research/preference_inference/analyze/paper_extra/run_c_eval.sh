#!/bin/bash
# Paper extra C evaluation (log Sec.4.3, 14.2): the full set for one trained run. Kusano-san asked to start it
# automatically when training ends (log Sec.18). Run INSIDE the container: GPU=<g> bash run_c_eval.sh <signrand|pzero> <seed>
set -u
cd "$(dirname "$0")/../.."
c=$1; s=$2; GPU=${GPU:-6}
export CUDA_VISIBLE_DEVICES=$GPU
d=$([ $c = signrand ] && echo C_signrand || echo C_pzero)
E=paper_extra/C_signrand/eval
L=data/result/$E/logs; mkdir -p $L
C="python -u analyze/paper_extra/c_run.py"
EPS="160 170 180 190 200"
$C trial --exp_fmt paper_extra/$d/s1_{}_mse --conds $c --seed $s --epochs $EPS --no_p2 --out ../$E/s1eval/$c/s$s > $L/trial_${c}_s$s.log 2>&1
$C trial --exp_fmt paper_extra/$d/s1_{}_mse --conds $c --seed $s --epochs 200 --no_p2 --tag _rep --out ../$E/s1eval/$c/s$s > $L/trial_rep_${c}_s$s.log 2>&1
G3M="$c:$s"; [ $c = signrand ] && G3M="$G3M mixed80:$s random:$s"
$C g3flip --models $G3M > $L/g3flip_${c}_s$s.log 2>&1
for e in $EPS; do
    $C g5 --exp_fmt paper_extra/$d/s1_{}_l1_s$s --conds $c --seed $s --epoch $e --sub ../$E/g5/$c/s$s > $L/g5_${c}_s${s}_e$e.log 2>&1
done
$C g5 --exp_fmt paper_extra/$d/s1_{}_l1_s$s --conds $c --seed $s --epoch 200 --sub ../$E/g5_rep/$c/s$s > $L/g5_rep_${c}_s$s.log 2>&1
$C decomp --seed $s --conds $c > $L/decomp_${c}_s$s.log 2>&1
$C d6 --models $c:$s --sets green pref --out s$s > $L/d6_${c}_s$s.log 2>&1
f=$([ $c = signrand ] && echo csr || echo cpz)
$C inverse --models $f:$c:$s --epochs $EPS --sets pref --out ../paper_extra/C_signrand/eval/inverse > $L/inverse_${c}_s$s.log 2>&1
$C inverse --models ${f}mse:$c:$s --epochs $EPS --sets pref --out ../paper_extra/C_signrand/eval/inverse > $L/inverse_mse_${c}_s$s.log 2>&1
echo "tracebacks $(grep -l Traceback $L/*_${c}_s$s*.log | wc -l)" > $L/${c}_s$s.done
