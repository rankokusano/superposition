#!/bin/bash
set -u; cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES; export DEVICE=cuda:5
echo "[v5r4] $(date -u) start train 400ep"
python train.py --exp_config v5_r4_move --seed 0 --device cuda:5 > logs/v5_r4_move_train.log 2>&1
echo "[v5r4] train rc=$? last=$(ls data/result/v5_r4_move/0/model/ 2>/dev/null | tail -1)"
python analyze/check_convergence.py v5_r4_move --seeds 0 2>&1 | grep -v "^saved:"
echo "[v5r4] DONE $(date -u)"
