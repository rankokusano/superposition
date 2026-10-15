#!/bin/bash
# v10: B4-criterion baseline (user request 2026-10-15): layout read-out from
# R3-A_400 h1 (seed0, seed1; late-5 ep360-400) on the test splits of
# r2_a1random_a2rl + v10_swap_r2 (episode-paired -> --pair_groups).
# Outputs only under data/result/v10_swap/b4_baseline/.
set -e
cd /work/my_research/preference_inference
EXP=r3_a_direct_1000pretrain_400
DEV=${DEV:-cuda:7}
R=data/result/v10_swap/b4_baseline/${EXP}
for S in 0 1; do
  for E in 360 370 380 390 400; do
    J=$R/s$S/ep$E/layout_readout.json
    [ -f $J ] && continue
    for DS in r2_a1random_a2rl v10_swap_r2; do
      python -u analyze/v10/test_v10.py --exp_config $EXP --seed $S --device $DEV --test_epoch $E \
        --test_data_name $DS --test_batch_size 10 --test_modes test --eval_seed 0 --cudnn_deterministic \
        --save_targets state --out_dir $R/s$S/ep$E/$DS 2>&1 | grep -v 'it/s\]' | grep -v -i warn
    done
    python analyze/v10/layout_readout_v10.py --exp_config $EXP --seed $S --epoch $E \
      --datasets r2_a1random_a2rl v10_swap_r2 --run_dirs $R/s$S/ep$E/r2_a1random_a2rl $R/s$S/ep$E/v10_swap_r2 \
      --mode test --pair_groups --device $DEV --out $J
    [ -f $J ] && rm -f $R/s$S/ep$E/*/save/saved.h5
  done
done
echo B4_BASELINE_DONE
