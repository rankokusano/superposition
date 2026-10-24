#!/bin/bash
# v13 addition (docs/v13_experiment_log.md Sec.31): evaluation of an extra pair s (4 or 5) once its three stage-1
# chains are done: the v13 pair (no experience + 80%) and the v12 pair (no experience + 50%), same measures as the
# existing pairs. Usage INSIDE the container (detached): GPU=k bash run_extra_eval_v13.sh <seed>
set -u
cd "$(dirname "$0")"
s=$1; GPU=${GPU:-1}
case $GPU in 1|2|3|5|6|7) ;; *) echo "GPU $GPU is not allowed"; exit 1 ;; esac
export CUDA_VISIBLE_DEVICES=$GPU
L=data/result/v13/logs/extra; mkdir -p $L
until [ -f data/result/v12_s1_random_l1_s$s/chain_seed$s.done ] && [ -f data/result/v12_s1_mixed_l1_s$s/chain_seed$s.done ] && \
      [ -f data/result/v13_s1_mixed_l1_s$s/chain_seed$s.done ]; do sleep 300; done
{ for d in v12_s1_random v12_s1_mixed v13_s1_mixed; do
    echo "s$s $d: chain $(cat data/result/${d}_l1_s$s/chain_seed$s.done) mse $(ls data/result/${d}_mse/$s/model | wc -l) l1 $(ls data/result/${d}_l1_s$s/$s/model | wc -l) tb $(cat data/result/${d}_mse/train_seed$s.log data/result/${d}_l1_s$s/train_seed$s.log | grep -ac Traceback) $(date)"
  done; } >> $L/completion.log
R="python -u analyze/v13/v12run.py"
EPS="160 170 180 190 200"
step() { echo "s$s $1 exit $2 $(date)" >> $L/steps.log; }
# self side (mse late-5): v13 pair (80%) with the ep200 repeat, v12 pair (50%)
$R trial --exp_fmt v13_s1_{}_mse --seed $s --epochs $EPS --no_p2 --out ../v13/main/s1eval/s$s > $L/A13_s$s.log 2>&1; step A13 $?
$R trial --exp_fmt v13_s1_{}_mse --seed $s --epochs 200 --no_p2 --tag _rep --out ../v13/main/s1eval/s$s > $L/A13rep_s$s.log 2>&1; step A13rep $?
$R trial --exp_fmt v13ref_s1_{}_mse --seed $s --epochs $EPS --no_p2 --out ../v13/v12ref/s1eval/s$s > $L/A12_s$s.log 2>&1; step A12 $?
# h2 -> A-2 position (mse ep200)
$R h2 --runs random:$s:200 mixed:$s:200 --sub ../v13/main/h2_check_s$s > $L/h2_13_s$s.log 2>&1; step h2_13 $?
V13_REF=1 $R h2 --runs mixed:$s:200 --sub ../v13/v12ref/h2_check_s$s > $L/h2_12_s$s.log 2>&1; step h2_12 $?
# G5a / G5b (l1 late-5): v13 pair; v12 50% experienced (the no-experience model is the same)
for e in $EPS; do
  $R g5 --exp_fmt v13_s1_{}_l1_s$s --seed $s --epoch $e --sub ../v13/main/g5/s$s > $L/G5_13_s${s}_e$e.log 2>&1; step G5_13_e$e $?
  $R g5 --exp_fmt v13ref_s1_{}_l1_s$s --seed $s --epoch $e --conds mixed --sub ../v13/v12ref/g5/s$s > $L/G5_12_s${s}_e$e.log 2>&1; step G5_12_e$e $?
done
# sign-following part (no experience, 50%, 80%)
python -u analyze/v13/g5a_decomp_v13.py --seed $s > $L/decomp_s$s.log 2>&1; step decomp $?
# D6-like secondary (ep200)
$R d6 --exp_fmt v13_s1_{}_l1_s$s --seed $s --epoch 200 --sub ../v13/main/d6/s$s > $L/D6_13_s$s.log 2>&1; step D6_13 $?
$R d6 --exp_fmt v13ref_s1_{}_l1_s$s --seed $s --epoch 200 --sub ../v13/v12ref/d6/s$s > $L/D6_12_s$s.log 2>&1; step D6_12 $?
echo "s$s all done $(date)" >> $L/steps.log
