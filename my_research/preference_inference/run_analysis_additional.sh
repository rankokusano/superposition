#!/bin/bash
# Analyze additional validation test datasets.
# Prints RandomA2 and FixedA2 regression results for all 3 experiments.
# Usage: bash run_analysis_additional.sh [EPOCH] [SEED]
set -e

PI=/work/my_research/preference_inference
cd $PI

EPOCH=${1:-200}
SEED=${2:-0}

# ── RandomA2 regression ──────────────────────────────────────────────────────
echo "======================================================================"
echo "=== RandomA2 regression (cycle-order discrimination test) ==="
echo "======================================================================"

for EXP in new_exp_a new_exp_b_base new_exp_b_mg_ve; do
    echo ""
    echo "--- $EXP ---"
    SAVED_H5=data/result/${EXP}/${SEED}/test/test_self_rl_other_random/save
    cd analyze
    for LAYER in self other; do
        python regression.py \
            --result_dir ../$SAVED_H5 \
            --epoch $EPOCH \
            --mode eval \
            --layer $LAYER
    done
    cd ..
done

# ── FixedA2 joint 4-class regression ────────────────────────────────────────
echo ""
echo "======================================================================"
echo "=== FixedA2 joint regression (static preference test) ==="
echo "======================================================================"

for EXP in new_exp_a new_exp_b_base new_exp_b_mg_ve; do
    echo ""
    echo "--- $EXP ---"
    RESULT_BASE=data/result/${EXP}/${SEED}/test
    for LAYER in self other; do
        python analyze/regression_fixed.py \
            --result_base $RESULT_BASE \
            --epoch $EPOCH \
            --mode eval \
            --layer $LAYER
    done
done

echo ""
echo "=== Additional analysis done ==="
echo ""
echo "Decision guide:"
echo "  RandomA2 other-layer accuracy >= 95%  -> preference encoding confirmed"
echo "  RandomA2 other-layer accuracy <  70%  -> cycle pattern suspected -> run Step 4"
echo "  FixedA2  other-layer accuracy >= 90%  -> static preference readable from h2"
