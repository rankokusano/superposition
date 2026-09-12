#!/bin/bash
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:6
echo "[fig5v5-2] $(date -u) train Autoencoder fig5_v5_base 100ep"
python train.py --exp_config fig5_v5_base --seed 0 --device cuda:6 > logs/fig5_v5_train2.log 2>&1
echo "[fig5v5-2] train rc=$? last=$(ls data/result/fig5_v5_base/0/model/ 2>/dev/null | tail -1)"
[ -f data/result/fig5_v5_base/0/model/00100.pth ] || { echo "[fig5v5-2] no ep100 -- abort"; exit 1; }
df -h /work | tail -1
python test.py --exp_config fig5_v5_base --seed 0 --device cuda:6 \
  --test_epoch 100 --test_data_name grid --test_modes eval \
  --save_targets self_vision other_vision self_position other_position > logs/fig5_v5_test2.log 2>&1
echo "[fig5v5-2] test rc=$?"
d=data/result/fig5_v5_base/0/test/grid/save
if [ -f "$d/saved.h5" ]; then
  ( cd "$d" && python /work/my_research/preference_inference/analyze/analyze_vpt.py --exp_config fig5_v5_base --epoch 100 ) > logs/fig5_v5_vpt2.log 2>&1
  echo "=== fig5_v5 RESULT ==="; head -6 "$d/vpt/0/histogram/result.txt"
fi
echo "[fig5v5-2] DONE $(date -u)"
