#!/bin/bash
set -u
cd /work/my_research/preference_inference
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda:0
echo "[v5-chain] $(date -u) starting v5_base_mse (400ep, scratch) on GPU0"
python train.py --exp_config v5_base_mse --seed 0 --device cuda:0 > logs/v5_base_mse.log 2>&1
rc=$?
echo "[v5-chain] $(date -u) v5_base_mse exit rc=$rc  last ckpt: $(ls data/result/v5_base_mse/0/model/ 2>/dev/null | tail -1)"
if [ $rc -ne 0 ] || [ ! -f data/result/v5_base_mse/0/model/00400.pth ]; then
  echo "[v5-chain] v5_base_mse did not reach ep400 -- NOT starting l1 stage"; exit 1
fi
echo "[v5-chain] $(date -u) starting v5_base_l1 (200ep, pretrain=v5_base_mse ep400)"
python train.py --exp_config v5_base_l1 --seed 0 --device cuda:0 > logs/v5_base_l1.log 2>&1
echo "[v5-chain] $(date -u) v5_base_l1 exit rc=$?  last ckpt: $(ls data/result/v5_base_l1/0/model/ 2>/dev/null | tail -1)"
echo "[v5-chain] DONE"
