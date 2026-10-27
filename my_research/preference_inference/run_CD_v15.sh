#!/bin/bash
# v15 Sec.33 (C) (B) v15i seeds 1-2 and (D) NS control v15i exp/noexp seeds 0-2, 2(b) design, trial scale.
# Four slots (two per GPU); each slot runs its chains in order: phase 1 (mse 150) -> phase 2 hidden (150) -> GPU eval.
# A chain is not started if data/result/v15/STOP_NEW exists. Args: two GPU ids. Run INSIDE the container, detached.
set -u
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs; mkdir -p $L
G1=$1; G2=$2
chain() { local p1=$1 p2=$2 seed=$3 gpu=$4 tr=$5 ev=$6
  if [ -f data/result/v15/STOP_NEW ]; then echo "$p2 seed $seed not started (STOP_NEW) $(date)" >> $L/done_CD; return; fi
  for exp in $p1 $p2; do mkdir -p data/result/$exp
    CUDA_VISIBLE_DEVICES=$gpu python -u $tr --exp_config $exp --seed $seed --device cuda:0 --cudnn_deterministic \
      --data_load_memory > data/result/$exp/train_seed$seed.log 2>&1
    rc=$?; echo "exit $rc" > data/result/$exp/done_seed$seed; echo "$exp seed $seed train exit $rc $(date)" >> $L/done_CD
    [ $rc = 0 ] || return
  done
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 nice -n 10 python -u $ev --exp $p2 --seed $seed --device cuda:0 \
    > $L/hide_${p2}_seed$seed.log 2>&1
  rc=$?; echo "$p2 seed $seed eval exit $rc $(date)" >> $L/done_CD
}
SH="train_v15.py analyze/v15/eval_s2_hide_v15.py"; NS="train_v15ns.py analyze/v15/eval_s2_hide_ns_v15.py"
# round 1: (C) seeds 1, 2 and NS seed 0; round 2: NS seeds 1, 2
( chain v15_s2p1_expY_v15i_mse v15_s2p2h_expY_v15i_s1 1 $G1 $SH; chain v15_s2p1_exp_v15i_ns_mse v15_s2p2h_exp_v15i_ns_s1 1 $G1 $NS ) &
( chain v15_s2p1_exp_v15i_ns_mse v15_s2p2h_exp_v15i_ns 0 $G1 $NS; chain v15_s2p1_noexp_v15i_ns_mse v15_s2p2h_noexp_v15i_ns_s1 1 $G1 $NS ) &
( chain v15_s2p1_expY_v15i_mse v15_s2p2h_expY_v15i_s2 2 $G2 $SH; chain v15_s2p1_exp_v15i_ns_mse v15_s2p2h_exp_v15i_ns_s2 2 $G2 $NS ) &
( chain v15_s2p1_noexp_v15i_ns_mse v15_s2p2h_noexp_v15i_ns 0 $G2 $NS; chain v15_s2p1_noexp_v15i_ns_mse v15_s2p2h_noexp_v15i_ns_s2 2 $G2 $NS ) &
wait
echo ALL_DONE_CD >> $L/done_CD
