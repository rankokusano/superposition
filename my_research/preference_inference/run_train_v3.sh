#!/bin/bash
# v3 3-stage training pipeline.
# Run from /work/my_research/preference_inference inside Docker.
#
# Usage:
#   bash run_train_v3.sh [--seed N] [--stage 1|2|3]
#
# Stages:
#   1: B-base MSE   (v3_exp_b_base_mse)  -- no pretrain, all params free
#   2: B-base L1    (v3_exp_b_base_l1)   -- loads stage1, SM+enc+int+FPM frozen
#   3: B-MGVE       (v3_exp_b_mgve)      -- loads stage2, SM+enc+dec+int+FPM frozen

set -e

SEED=0
START_STAGE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)   SEED="$2";   shift 2 ;;
        --stage)  START_STAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

TRAIN="python train.py --seed $SEED --device cuda:0"

echo "=========================================="
echo " v3 training pipeline  seed=$SEED"
echo " Starting from stage $START_STAGE"
echo "=========================================="

# ── Stage 1: B-base MSE ────────────────────────────────────────────────
if [ "$START_STAGE" -le 1 ]; then
    echo ""
    echo "[Stage 1] B-base MSE (v3_exp_b_base_mse)"
    $TRAIN --exp_config v3_exp_b_base_mse
    echo "[Stage 1] done."
fi

# ── Stage 2: B-base L1 ─────────────────────────────────────────────────
if [ "$START_STAGE" -le 2 ]; then
    echo ""
    echo "[Stage 2] B-base L1 (v3_exp_b_base_l1)"
    $TRAIN --exp_config v3_exp_b_base_l1
    echo "[Stage 2] done."
fi

# ── Stage 3: B-MGVE ────────────────────────────────────────────────────
if [ "$START_STAGE" -le 3 ]; then
    echo ""
    echo "[Stage 3] B-MGVE (v3_exp_b_mgve)"
    $TRAIN --exp_config v3_exp_b_mgve
    echo "[Stage 3] done."
fi

echo ""
echo "=========================================="
echo " All stages complete."
echo "=========================================="
