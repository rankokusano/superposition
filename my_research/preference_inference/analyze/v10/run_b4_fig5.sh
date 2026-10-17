#!/bin/bash
# v10 B4 Fig.5 + prediction images for one decoder config (Sec.5.30).
#   bash analyze/v10/run_b4_fig5.sh <fig5_exp> <b3_exp> "<b3 seeds served>" <device>
# 1. train the Autoencoder decoder (100ep, --seed 0) -> data/result/<fig5_exp>/0/
# 2. grid (original) and v10_swap_grid at ep100, margin 1 -> b4/fig5/<fig5_exp>_ep100_{grid,v10_swap_grid}.json
# 3. prediction images on v10_swapmix_r2_viz20 for each served B3 seed (ep400)
set -u
FX=$1; BX=$2; SEEDS=$3; DEV=$4
S=data/result/v10_swap/b4
mkdir -p $S/fig5 $S/logs
[ -f data/result/$FX/0/model/00100.pth ] || python -u train.py --exp_config $FX --seed 0 --device $DEV > $S/logs/train_$FX.log 2>&1
for DS in grid v10_swap_grid; do
  J=$S/fig5/${FX}_ep100_${DS}.json
  [ -f $J ] && continue
  O=$S/fig5/run_${FX}_$DS
  python -u analyze/v10/test_v10.py --exp_config $FX --seed 0 --device $DEV --test_epoch 100 --test_data_name $DS \
    --test_batch_size 100 --test_modes eval --save_targets self_vision other_vision self_position other_position \
    --out_dir $O > $S/logs/fig5_${FX}_$DS.log 2>&1
  python analyze/v10/vpt_metrics_v10.py --data_h5 data/data/$DS/data.h5 --saved_h5 $O/save/saved.h5 --epoch 100 \
    --margin 1 --label ${FX}_ep100_$DS --out $J >> $S/logs/fig5_${FX}_$DS.log 2>&1 && rm -f $O/save/saved.h5
done
P=$S/pred_images
A=$P/ae_${FX}
[ -f $A/save/saved.h5 ] || python -u analyze/v10/test_v10.py --exp_config $FX --seed 0 --device $DEV --test_epoch 100 \
  --test_data_name v10_swapmix_r2_viz20 --test_batch_size 100 --test_modes eval --save_targets self_vision other_vision \
  --out_dir $A > $S/logs/predimg_ae_$FX.log 2>&1
for SD in $SEEDS; do
  M=$P/${BX}_s${SD}_ep400
  [ -f $M/model/save/saved.h5 ] || python -u analyze/v10/test_v10.py --exp_config $BX --seed $SD --device $DEV --test_epoch 400 \
    --test_data_name v10_swapmix_r2_viz20 --test_batch_size 10 --test_modes eval --eval_seed 0 --cudnn_deterministic \
    --save_targets self_vision state --out_dir $M/model > $S/logs/predimg_${BX}_s$SD.log 2>&1
  for spec in "original 0 4" "swap_rg 1 2"; do
    set -- $spec
    python analyze/v10/pred_images_v10.py --label "$BX s$SD ep400 + $FX ep100 (viz20 episodes of layout $1)" \
      --model_epoch 400 --ae_epoch 100 --episodes $2 $3 --layouts $1 \
      --model_h5 $M/model/save/saved.h5 --ae_h5 $A/save/saved.h5 \
      --data_h5 data/data/v10_swapmix_r2_viz20/data.h5 --out_dir $M/$1 >> $S/logs/predimg_${BX}_s$SD.log 2>&1
  done
  [ -f $M/swap_rg/pred_images_arrays.npz ] && rm -f $M/model/save/saved.h5
done
echo "FIG5_DONE $FX"
