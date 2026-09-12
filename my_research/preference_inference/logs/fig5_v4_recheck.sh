#!/bin/bash
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:4
echo "[recheck] $(date -u) retrain fig5_v4_r3stay400 Autoencoder (100ep) with fixed loader"
python train.py --exp_config fig5_v4_r3stay400 --seed 0 --device cuda:4 > logs/fig5_v4_recheck_train.log 2>&1
echo "[recheck] train rc=$?  last=$(ls data/result/fig5_v4_r3stay400/0/model/ 2>/dev/null | tail -1)"
python test.py --exp_config fig5_v4_r3stay400 --seed 0 --device cuda:4 \
  --test_epoch 100 --test_data_name grid --test_modes eval \
  --save_targets self_vision other_vision self_position other_position \
  --test_name grid_recheck > logs/fig5_v4_recheck_test.log 2>&1
echo "[recheck] test rc=$?"
d=data/result/fig5_v4_r3stay400/0/test/grid_recheck/save
if [ -f "$d/saved.h5" ]; then
  ( cd "$d" && python /work/my_research/preference_inference/analyze/analyze_vpt.py --exp_config fig5_v4_r3stay400 --epoch 100 ) > logs/fig5_v4_recheck_vpt.log 2>&1
  echo "=== RECHECK RESULT (expect other_true_and_other_rec ~0.058 < other_true_and_self_rec ~0.149) ==="
  head -6 "$d/vpt/0/histogram/result.txt" 2>/dev/null
fi
echo "[recheck] DONE $(date -u)"
