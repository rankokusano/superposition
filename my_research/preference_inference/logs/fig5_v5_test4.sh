#!/bin/bash
set -u; cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES; export DEVICE=cuda:6
echo "[t4] $(date -u) grid test"
python test.py --exp_config fig5_v5_base --seed 0 --device cuda:6 \
  --test_epoch 100 --test_data_name grid --test_batch_size 100 --test_modes eval \
  --save_targets self_vision other_vision self_position other_position > logs/fig5_v5_test4.log 2>&1
echo "[t4] test rc=$?"
d=data/result/fig5_v5_base/0/test/grid/save
[ -f "$d/saved.h5" ] || { echo "[t4] NO saved.h5"; tail -15 logs/fig5_v5_test4.log; exit 1; }
for M in 1 0; do
  ( cd "$d" && python /work/my_research/preference_inference/analyze/analyze_vpt.py --epoch 100 --margin $M ) > logs/fig5_v5_vpt4_m${M}.log 2>&1
  echo "[t4] vpt margin=$M rc=$?"
  cat "$d/vpt/$M/histogram/result.txt" 2>/dev/null | head -6
done
echo "[t4] keeping saved.h5 until result confirmed. size: $(du -h $d/saved.h5|cut -f1)"
echo "[t4] DONE $(date -u)"
