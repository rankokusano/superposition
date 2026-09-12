#!/bin/bash
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:4
export DEVICE_HARNESS=cuda:4
echo "[paper@r3stay] $(date -u) test exp1_l1_1000 ep200 on r3_stay"
python test.py --exp_config exp1_l1_1000 --seed 0 --device cuda:4 \
  --test_epoch 200 --test_data_name r3_stay --test_modes eval \
  --save_targets state self_position other_position --test_name r3stay_probe \
  --eval_seed 0 --cudnn_deterministic > logs/paper_on_r3stay_test.log 2>&1
echo "[paper@r3stay] test rc=$?"
python analyze/regression_baseline_v4.py \
  --saved_h5 data/result/exp1_l1_1000/0/test/r3stay_probe/save/saved.h5 \
  --epoch 200 --label paper_exp1l1000_on_r3stay > logs/paper_on_r3stay_r2.log 2>&1 || \
python analyze/regression_baseline_v4.py exp1_l1_1000 200 r3stay_probe > logs/paper_on_r3stay_r2.log 2>&1
echo "[paper@r3stay] r2 rc=$?"
grep -E "h1 ->|h2 ->|R\^2" logs/paper_on_r3stay_r2.log
echo "[paper@r3stay] DONE"
