#!/bin/bash
# v10 stage A chain (inside the container, from /work/my_research/preference_inference).
# Waits for the two collections, then runs everything of stage A. GPU0 is never used.
set -u
cd /work/my_research/preference_inference
L=data/result/v10_swap/stageA/logs
S=data/result/v10_swap/stageA

wait_for() { until grep -q '^Saved:' "$1" 2>/dev/null; do sleep 60; done; }

echo "[chain] $(date) waiting for v10_swap_r2"
wait_for $L/collect_v10_swap_r2.log
python analyze/v10/make_viz20.py --src data/data/v10_swap_r2/data.h5 --dst data/data/v10_swap_r2_viz20/data.h5 --n 20
python analyze/v10/a2_terminal_v10.py > $L/a2_terminal.log 2>&1

echo "[chain] $(date) late-5 evals"
bash analyze/v10/run_stageA_eval.sh r3_a_direct_1000pretrain_400 0 cuda:3 "360 370 380 390 400" > $L/eval_r3a_s0.log 2>&1 &
P1=$!
bash analyze/v10/run_stageA_eval.sh r3_a_direct_1000pretrain_400 1 cuda:4 "360 370 380 390 400" > $L/eval_r3a_s1.log 2>&1 &
P2=$!
bash analyze/v10/run_stageA_eval.sh v5_base_l1 0 cuda:5 "160 170 180 190 200" > $L/eval_v5_s0.log 2>&1 &
P3=$!

# prediction images (viz20), final checkpoint of each model; Fig.5 AE on viz20
for spec in "r3_a_direct_1000pretrain_400 0 400 fig5_v4_exp1l1000" "r3_a_direct_1000pretrain_400 1 400 fig5_v4_exp1l1000" "v5_base_l1 0 200 fig5_v5_base"; do
  set -- $spec; EXP=$1; SEED=$2; E=$3; AE=$4
  D=$S/pred_images/${EXP}_s${SEED}_ep${E}
  for C in original swap_rg; do
    if [ $C = original ]; then DS=r2_a1random_a2rl_viz20; else DS=v10_swap_r2_viz20; fi
    python -u analyze/v10/test_v10.py --exp_config $EXP --seed $SEED --device cuda:6 --test_epoch $E \
      --test_data_name $DS --test_batch_size 10 --test_modes eval --eval_seed 0 --cudnn_deterministic \
      --save_targets self_vision state --out_dir $D/model_$C > $L/predimg_${EXP}_s${SEED}_$C.log 2>&1
    [ -f $S/pred_images/ae_${AE}_ep100/$C/save/saved.h5 ] || \
    python -u analyze/v10/test_v10.py --exp_config $AE --seed 0 --device cuda:6 --test_epoch 100 \
      --test_data_name $DS --test_batch_size 100 --test_modes eval \
      --save_targets self_vision other_vision --out_dir $S/pred_images/ae_${AE}_ep100/$C > $L/predimg_ae_${AE}_$C.log 2>&1
  done
  python analyze/v10/pred_images_v10.py --label "${EXP} s${SEED} ep${E} + ${AE} ep100" --model_epoch $E --ae_epoch 100 \
    --model_h5 $D/model_original/save/saved.h5 $D/model_swap_rg/save/saved.h5 \
    --ae_h5 $S/pred_images/ae_${AE}_ep100/original/save/saved.h5 $S/pred_images/ae_${AE}_ep100/swap_rg/save/saved.h5 \
    --data_h5 data/data/r2_a1random_a2rl_viz20/data.h5 data/data/v10_swap_r2_viz20/data.h5 \
    --out_dir $D > $L/predimg_plot_${EXP}_s${SEED}.log 2>&1
done

echo "[chain] $(date) waiting for v10_swap_grid"
until grep -q '^Saved:' $L/collect_v10_swap_grid.log 2>/dev/null; do sleep 60; done
for AE in fig5_v4_exp1l1000 fig5_v5_base; do
  for C in A0 A1; do
    if [ $C = A0 ]; then DS=grid; else DS=v10_swap_grid; fi
    J=$S/fig5/${AE}_ep100_${C}.json
    [ -f $J ] && continue
    O=$S/fig5/${AE}_ep100/$C
    python -u analyze/v10/test_v10.py --exp_config $AE --seed 0 --device cuda:6 --test_epoch 100 \
      --test_data_name $DS --test_batch_size 100 --test_modes eval \
      --save_targets self_vision other_vision self_position other_position --out_dir $O > $L/fig5_${AE}_$C.log 2>&1
    python analyze/v10/vpt_metrics_v10.py --data_h5 data/data/$DS/data.h5 --saved_h5 $O/save/saved.h5 \
      --epoch 100 --margin 1 --label ${AE}_ep100_$C --out $J >> $L/fig5_${AE}_$C.log 2>&1
    [ -f $J ] && rm -f $O/save/saved.h5
  done
done

wait $P1 $P2 $P3
python analyze/v10/aggregate_stageA.py --exp_config r3_a_direct_1000pretrain_400 --seeds 0 1 --epochs 360 370 380 390 400 > $L/aggregate_r3a.log 2>&1
python analyze/v10/aggregate_stageA.py --exp_config v5_base_l1 --seeds 0 --epochs 160 170 180 190 200 > $L/aggregate_v5.log 2>&1
echo "[chain] $(date) ALL DONE"
