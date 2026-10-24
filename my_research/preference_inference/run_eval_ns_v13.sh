#!/bin/bash
# v13-NS stage-1 evaluation of one pair (docs/v13_experiment_log.md Sec.19.3, 26): the v13 measures on the NS pair
# (no experience v13ns_s1_random_*, experienced 80% v13ns_s1_mixed_*), via analyze/v13/v12run.py (NS class registered).
#   GPU=k bash run_eval_ns_v13.sh <seed>      (GPU 1, 2, 6 or 7)
set -u
cd "$(dirname "$0")"
s=$1; GPU=${GPU:-7}
case $GPU in 1|2|6|7) ;; *) echo "GPU $GPU is not allowed"; exit 1 ;; esac
export CUDA_VISIBLE_DEVICES=$GPU
L=data/result/v13/logs/eval_ns; mkdir -p $L
R="python -u analyze/v13/v12run.py"
EPS="160 170 180 190 200"
echo "s$s: tracebacks $(cat data/result/v13ns_s1_*_mse/train_seed$s.log data/result/v13ns_s1_*_l1_s$s/train_seed$s.log | grep -ac Traceback) at $(date)" >> $L/completion.log
# self side: G1-G3 (mse late-5) and the ep200 repeat
$R trial --exp_fmt v13ns_s1_{}_mse --seed $s --epochs $EPS --no_p2 --out ../v13/ns/s1eval/s$s > $L/A_s$s.log 2>&1
$R trial --exp_fmt v13ns_s1_{}_mse --seed $s --epochs 200 --no_p2 --tag _rep --out ../v13/ns/s1eval/s$s > $L/A_rep_s$s.log 2>&1
# process-2 side: h2 -> A-2 position (selection criterion as a check)
V13_NS=1 $R h2 --runs random:$s:200 mixed:$s:200 --sub ../v13/ns/h2_check_s$s > $L/h2_s$s.log 2>&1
# G5a / G5b (l1 late-5) with per-episode values; D6-like secondary at ep200
for e in $EPS; do
  $R g5 --exp_fmt v13ns_s1_{}_l1_s$s --seed $s --epoch $e --sub ../v13/ns/g5/s$s > $L/G5_s${s}_e$e.log 2>&1
done
$R d6 --exp_fmt v13ns_s1_{}_l1_s$s --seed $s --epoch 200 --sub ../v13/ns/d6/s$s > $L/D6_s$s.log 2>&1
# sign-following part of G5a (Sec.15 (b))
python -u analyze/v13/g5a_decomp_v13.py --ns --seed $s > $L/decomp_s$s.log 2>&1
echo "exit $?" > $L/PAIR_s$s.done
