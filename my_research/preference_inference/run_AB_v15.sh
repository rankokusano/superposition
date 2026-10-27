#!/bin/bash
# v15 Sec.28.3 / 30 (A) seeds 1-2 and (B) rank-Y experienced, 2(b) design (A-2 hidden in phase 2), trial scale.
# Six slots (two per GPU); each slot runs its chains in order: phase 1 (mse 150) -> phase 2 hidden (150) -> eval_s2_hide_v15.
# A chain is not started if data/result/v15/STOP_NEW exists (Sec.30.5). Args: three GPU ids. Run INSIDE the container, detached.
set -u
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs; mkdir -p $L
G1=$1; G2=$2; G3=$3
chain() { local p1=$1 p2=$2 seed=$3 gpu=$4
  if [ -f data/result/v15/STOP_NEW ]; then echo "$p2 seed $seed not started (STOP_NEW) $(date)" >> $L/done_AB; return; fi
  for exp in $p1 $p2; do mkdir -p data/result/$exp
    CUDA_VISIBLE_DEVICES=$gpu python -u train_v15.py --exp_config $exp --seed $seed --device cuda:0 --cudnn_deterministic \
      --data_load_memory > data/result/$exp/train_seed$seed.log 2>&1
    rc=$?; echo "exit $rc" > data/result/$exp/done_seed$seed; echo "$exp seed $seed train exit $rc $(date)" >> $L/done_AB
    [ $rc = 0 ] || return
  done
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s2_hide_v15.py --exp $p2 --seed $seed --device cuda:0 \
    > $L/hide_${p2}_seed$seed.log 2>&1
  rc=$?; echo "$p2 seed $seed eval exit $rc $(date)" >> $L/done_AB
}
# slot lists (order = priority: (B) first, then (A) v4, then (A) v15i)
( chain v15_s2p1_expY_v4_mse v15_s2p2h_expY_v4 0 $G1; chain v15_s2p1_exp_v15i_mse v15_s2p2h_exp_v15i_s1 1 $G1 ) &
( chain v15_s2p1_exp_v4_mse v15_s2p2h_exp_v4_s1 1 $G1; chain v15_s2p1_noexp_v15i_mse v15_s2p2h_noexp_v15i_s1 1 $G1 ) &
( chain v15_s2p1_expY_v15i_mse v15_s2p2h_expY_v15i 0 $G2; chain v15_s2p1_exp_v15i_mse v15_s2p2h_exp_v15i_s2 2 $G2 ) &
( chain v15_s2p1_noexp_v4_mse v15_s2p2h_noexp_v4_s1 1 $G2 ) &
( chain v15_s2p1_exp_v4_mse v15_s2p2h_exp_v4_s2 2 $G3; chain v15_s2p1_noexp_v15i_mse v15_s2p2h_noexp_v15i_s2 2 $G3 ) &
( chain v15_s2p1_noexp_v4_mse v15_s2p2h_noexp_v4_s2 2 $G3 ) &
wait
echo ALL_DONE_AB >> $L/done_AB
