#!/bin/bash
# v15 Sec.39 H4 (fewer observations): exp / noexp v15i, seeds 0-2, phase 2 only (2(b) design, G only) on the first 100 train
# episodes for 1500 epochs, then the GPU evaluation of the last checkpoint (ep1500). Six runs, two per GPU (exp and noexp of
# one seed on one card). A run is not started if data/result/v15/STOP_NEW exists. Args: three GPU ids. Run INSIDE the
# container, detached.
set -u
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs; mkdir -p $L
one() { local exp=$1 seed=$2 gpu=$3
  if [ -f data/result/v15/STOP_NEW ]; then echo "$exp seed $seed not started (STOP_NEW) $(date)" >> $L/done_H4; return; fi
  mkdir -p data/result/$exp
  CUDA_VISIBLE_DEVICES=$gpu python -u train_v15.py --exp_config $exp --seed $seed --device cuda:0 --cudnn_deterministic \
    --data_load_memory > data/result/$exp/train_seed$seed.log 2>&1
  rc=$?; echo "exit $rc" > data/result/$exp/done_seed$seed; echo "$exp seed $seed train exit $rc $(date)" >> $L/done_H4
  [ $rc = 0 ] || return
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s2_hide_v15.py --exp $exp --seed $seed \
    --epoch 1500 --device cuda:0 > $L/hide_${exp}_seed$seed.log 2>&1
  rc=$?; echo "$exp seed $seed eval exit $rc $(date)" >> $L/done_H4
}
G=($1 $2 $3)
for s in 0 1 2; do suf=$([ $s = 0 ] && echo "" || echo "_s$s")
  one v15_s2p2h_exp_v15i_o100$suf $s ${G[$s]} &
  one v15_s2p2h_noexp_v15i_o100$suf $s ${G[$s]} &
done
wait
echo ALL_DONE_H4 >> $L/done_H4
