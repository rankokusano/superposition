#!/bin/bash
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:6
echo "[fig5v5-t3] $(date -u) test on grid (batch 100)"
python test.py --exp_config fig5_v5_base --seed 0 --device cuda:6 \
  --test_epoch 100 --test_data_name grid --test_batch_size 100 --test_modes eval \
  --save_targets self_vision other_vision self_position other_position > logs/fig5_v5_test3.log 2>&1
echo "[fig5v5-t3] test rc=$?"
d=data/result/fig5_v5_base/0/test/grid/save
if [ -f "$d/saved.h5" ]; then
  ( cd "$d" && python /work/my_research/preference_inference/analyze/analyze_vpt.py --exp_config fig5_v5_base --epoch 100 ) > logs/fig5_v5_vpt3.log 2>&1
  echo "=== fig5_v5 RESULT ==="; head -6 "$d/vpt/0/histogram/result.txt"
  echo "--- cleanup 7GB grid saved.h5 (result.txt 取得済) ---"
  rm -f "$d/saved.h5"; df -h /work | tail -1
else
  echo "[fig5v5-t3] NO saved.h5"; tail -15 logs/fig5_v5_test3.log
fi
echo "[fig5v5-t3] DONE $(date -u)"
