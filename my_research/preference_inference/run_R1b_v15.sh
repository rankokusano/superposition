#!/bin/bash
# v15 Sec.39.1 R1b: phase 2 only from the start checkpoints of analyze/v15/make_r1b_v15.py, seeds 0-2, then the GPU
# evaluation. Started only when the R1 verdict (judge_R12_v15i_cpu.json) is 'pathway needed' or 'neither (describe)'.
# A run is not started if data/result/v15/STOP_NEW exists. Args: three GPU ids (seed 0, 1, 2). Run INSIDE the container, detached.
set -u
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs; mkdir -p $L
v=$(python -c "import json; print(json.load(open('data/result/v15/stage2/judge_R12_v15i_cpu.json'))['R1']['verdict'])")
case "$v" in "pathway needed"|"neither (describe)") ;; *) echo "R1b not run (R1 verdict: $v) $(date)" >> $L/done_R1b; exit 0;; esac
one() { local exp=$1 seed=$2 gpu=$3
  if [ -f data/result/v15/STOP_NEW ]; then echo "$exp seed $seed not started (STOP_NEW) $(date)" >> $L/done_R1b; return; fi
  mkdir -p data/result/$exp
  CUDA_VISIBLE_DEVICES=$gpu python -u train_v15ns.py --exp_config $exp --seed $seed --device cuda:0 --cudnn_deterministic \
    --data_load_memory > data/result/$exp/train_seed$seed.log 2>&1
  rc=$?; echo "exit $rc" > data/result/$exp/done_seed$seed; echo "$exp seed $seed train exit $rc $(date)" >> $L/done_R1b
  [ $rc = 0 ] || return
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s2_hide_ns_v15.py --exp $exp --seed $seed \
    --device cuda:0 > $L/hide_${exp}_seed$seed.log 2>&1
  rc=$?; echo "$exp seed $seed eval exit $rc $(date)" >> $L/done_R1b
}
G=($1 $2 $3)
for s in 0 1 2; do suf=$([ $s = 0 ] && echo "" || echo "_s$s")
  one v15_s2p2h_noexp_v15i_r1b$suf $s ${G[$s]} &
done
wait
echo ALL_DONE_R1B >> $L/done_R1b
