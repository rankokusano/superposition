#!/bin/bash
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:6
echo "[fig5-v5] $(date -u) train Autoencoder (fig5_v5_base, 100ep)"
python train.py --exp_config fig5_v5_base --seed 0 --device cuda:6 > logs/fig5_v5_train.log 2>&1
echo "[fig5-v5] train rc=$?  last=$(ls data/result/fig5_v5_base/0/model/ 2>/dev/null | tail -1)"
echo "[fig5-v5] $(date -u) test.py on grid -> saved.h5"
python test.py --exp_config fig5_v5_base --seed 0 --device cuda:6 \
  --test_epoch 100 --test_data_name grid --test_modes eval \
  --save_targets self_vision other_vision self_position other_position > logs/fig5_v5_test.log 2>&1
echo "[fig5-v5] test rc=$?"
d=data/result/fig5_v5_base/0/test/grid/save
if [ -f "$d/saved.h5" ]; then
  ( cd "$d" && python /work/my_research/preference_inference/analyze/analyze_vpt.py --exp_config fig5_v5_base --epoch 100 ) > logs/fig5_v5_vpt.log 2>&1 || \
  ( cd "$d" && python /work/analyze/analyze_vpt.py --exp_config fig5_v5_base --epoch 100 ) > logs/fig5_v5_vpt.log 2>&1
  echo "[fig5-v5] vpt rc=$?"
  echo "=== RESULT ==="
  cat "$d/vpt/0/histogram/result.txt" 2>/dev/null | head -6
else
  echo "[fig5-v5] NO saved.h5 -- test step failed"; tail -20 logs/fig5_v5_test.log
fi
echo "[fig5-v5] DONE $(date -u)"
