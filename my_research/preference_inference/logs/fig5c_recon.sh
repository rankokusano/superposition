#!/bin/bash
set -u; cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES; export DEVICE=cuda:6

gen() {  # exp
  for E in 10 100; do
    echo "[recon] $(date -u) $1 grid test ep$E"
    python test.py --exp_config "$1" --seed 0 --device cuda:6 \
      --test_epoch $E --test_data_name grid --test_batch_size 100 --test_modes eval \
      --save_targets self_vision other_vision self_position other_position \
      --test_name grid_recon > "logs/recon_${1}_ep${E}.log" 2>&1
    echo "[recon] $1 ep$E rc=$?"
  done
  d=data/result/$1/0/test/grid_recon/save
  ln -sf /work/my_research/preference_inference/data/data/grid/data.h5 "$d/data.h5"
  ln -sf /work/my_research/preference_inference/analyze "$d/analyze"
  for E in 10 100; do
    ( cd "$d" && python analyze/analyze_vpt.py --epoch $E --margin 1 ) > "logs/recon_vpt_${1}_ep${E}.log" 2>&1
    echo "[recon] === $1 ep$E vpt result ==="
    head -5 "$d/vpt/1/histogram/result.txt" 2>/dev/null
  done
  ls -la "$d/saved.h5"
}

gen fig5_v5_base
gen fig5_v4_r3stay400
df -h /work | tail -1
echo "[recon] DONE $(date -u)"
