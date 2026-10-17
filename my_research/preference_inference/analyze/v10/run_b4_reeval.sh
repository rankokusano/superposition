#!/bin/bash
# v10 confound-free re-evaluation (Sec.5.32) for one (exp_config, seed): late-5 checkpoints on the
# test split of v10_swapmix_eval_<tag>; outputs under data/result/v10_swap/b4_reeval/<tag>/eval/.
#   bash analyze/v10/run_b4_reeval.sh <tag> <exp_config> <seed> <device> "<epochs>"
set -u
TAG=$1; EXP=$2; SEED=$3; DEV=$4; EPOCHS=$5
DS=v10_swapmix_eval_$TAG
R=data/result/v10_swap/b4_reeval/$TAG/eval/$EXP/s$SEED
for E in $EPOCHS; do
  D=$R/ep$E; mkdir -p $D
  if [ ! -f $D/test_$TAG.json ]; then
    python -u analyze/v10/test_v10.py --exp_config $EXP --seed $SEED --device $DEV --test_epoch $E \
      --test_data_name $DS --test_batch_size 10 --test_modes test --eval_seed 0 --cudnn_deterministic \
      --save_targets state self_position self_vision --out_dir $D/run 2>&1 | grep -v 'it/s\]' | grep -v -i warn
    python analyze/v10/b4_eval_ckpt.py --exp_config $EXP --seed $SEED --epoch $E --run_dir $D/run \
      --device $DEV --dataset $DS --out_prefix $D/test_$TAG && rm -f $D/run/save/saved.h5
  fi
  echo "DONE $TAG $EXP s$SEED ep$E"
done
echo "ALL_DONE $TAG $EXP s$SEED"
