#!/bin/bash
# Paper round 11 E1 evaluation (docs/paper_extra_log.md Sec.35.4): for one trained 50% sign-randomised run, the set of
# instructions Sec.3.2 (as run_c_eval.sh for the 80% runs): G1-G3 and self error (mse late 5, ep200 twice), g3flip
# (hold / zero / flip, d_P) of signrand50 AND of the same pair's 50% run, G5a / G5b (l1 late 5, ep200 twice), the
# sign-following parts (decomp), and E2 (e2_run.py, signrand50, both stages, ep200 twice).
# Run INSIDE the container: GPU=<g> bash run_e1_eval.sh <seed>
set -u
cd "$(dirname "$0")/../.."
s=$1; export CUDA_VISIBLE_DEVICES=${GPU:?}
c=signrand50
E=paper_extra/E1_signrand50/eval
L=data/result/$E/logs; mkdir -p $L
C="python -u analyze/paper_extra/e1_run.py"
EPS="160 170 180 190 200"
$C trial --exp_fmt paper_extra/E1_signrand50/s1_{}_mse --conds $c --seed $s --epochs $EPS --no_p2 --out ../$E/s1eval/$c/s$s > $L/trial_${c}_s$s.log 2>&1
$C trial --exp_fmt paper_extra/E1_signrand50/s1_{}_mse --conds $c --seed $s --epochs 200 --no_p2 --tag _rep --out ../$E/s1eval/$c/s$s > $L/trial_rep_${c}_s$s.log 2>&1
$C g3flip --models $c:$s mixed50:$s > $L/g3flip_${c}_s$s.log 2>&1
for e in $EPS; do
    $C g5 --exp_fmt paper_extra/E1_signrand50/s1_{}_l1_s$s --conds $c --seed $s --epoch $e --sub ../$E/g5/$c/s$s > $L/g5_${c}_s${s}_e$e.log 2>&1
done
$C g5 --exp_fmt paper_extra/E1_signrand50/s1_{}_l1_s$s --conds $c --seed $s --epoch 200 --sub ../$E/g5_rep/$c/s$s > $L/g5_rep_${c}_s$s.log 2>&1
$C decomp --seed $s --conds $c > $L/decomp_${c}_s$s.log 2>&1
python -u analyze/paper_extra/e2_run.py --models sr50:$s > $L/e2_${c}_s$s.log 2>&1
python -u analyze/paper_extra/e2_run.py --models sr50:$s --epochs 200 --tag _rep > $L/e2_rep_${c}_s$s.log 2>&1
echo "tracebacks $(grep -l Traceback $L/*_${c}_s$s*.log | wc -l)" > $L/${c}_s$s.done
