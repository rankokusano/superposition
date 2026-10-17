#!/bin/bash
# v10 B4 for one (exp_config, seed): late-5 checkpoints, all outputs under
# data/result/v10_swap/b4/eval/<exp>/s<seed>/ep<E>/. saved.h5 removed after its JSON exists.
# Usage (container, /work/my_research/preference_inference):
#   bash analyze/v10/run_b4.sh <exp_config> <seed> <device> "<epochs>"
set -u
EXP=$1; SEED=$2; DEV=$3; EPOCHS=$4
R=data/result/v10_swap/b4/eval/$EXP/s$SEED
for E in $EPOCHS; do
  D=$R/ep$E; mkdir -p $D
  # 1. swapmix test split: read-outs, distance R^2, vision error per layout
  if [ ! -f $D/test_swapmix.json ]; then
    python -u analyze/v10/test_v10.py --exp_config $EXP --seed $SEED --device $DEV --test_epoch $E \
      --test_data_name v10_swapmix_r2 --test_batch_size 10 --test_modes test --eval_seed 0 --cudnn_deterministic \
      --save_targets state self_position self_vision --out_dir $D/run_test_swapmix 2>&1 | grep -v 'it/s\]' | grep -v -i warn
    python analyze/v10/b4_eval_ckpt.py --exp_config $EXP --seed $SEED --epoch $E --run_dir $D/run_test_swapmix \
      --device $DEV --out_prefix $D/test_swapmix && rm -f $D/run_test_swapmix/save/saved.h5
  fi
  # 2/3. 4-axis R^2: in-dist (swapmix train split, per layout) and common (r2 train split)
  for DS in v10_swapmix_r2 r2_a1random_a2rl; do
    J=$D/r2_${DS}.json
    [ -f $J ] && continue
    python -u analyze/v10/test_v10.py --exp_config $EXP --seed $SEED --device $DEV --test_epoch $E \
      --test_data_name $DS --test_batch_size 10 --test_modes eval --eval_seed 0 --cudnn_deterministic \
      --save_targets state self_position other_position --out_dir $D/run_$DS 2>&1 | grep -v 'it/s\]' | grep -v -i warn
    LF=""; [ $DS = v10_swapmix_r2 ] && LF="--layout_from v10_swapmix_r2"
    python analyze/v10/b4_r2_ckpt.py --run_dir $D/run_$DS --epoch $E --dataset $DS $LF --out $J && rm -f $D/run_$DS/save/saved.h5
  done
  echo "DONE $EXP s$SEED ep$E"
done
echo "ALL_DONE $EXP s$SEED"
