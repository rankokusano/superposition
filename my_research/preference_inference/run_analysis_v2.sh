#!/bin/bash
# Run all v2 analysis: FixedA2 regression + Q̂² by preference.
#
# Run inside Docker:
#   docker exec kusano_research bash -c "cd /work/my_research/preference_inference && bash run_analysis_v2.sh [EPOCH] [SEED]"
set -e

PI=/work/my_research/preference_inference
cd $PI

EPOCH=${1:-200}
SEED=${2:-0}

echo "======================================================================"
echo "=== v2 Analysis started at: $(date) ==="
echo "======================================================================"

# ── FixedA2 regression (h² encoding static preference) ───────────────────────
echo ""
echo "=== v2 FixedA2 regression (h² layer preference encoding) ==="
for EXP in v2_exp_b_base v2_exp_b_mg_ve; do
    echo ""
    echo "--- $EXP ---"
    RESULT_BASE=data/result/${EXP}/${SEED}/test
    for LAYER in self other; do
        python analyze/regression_fixed_v2.py \
            --result_base $RESULT_BASE \
            --epoch $EPOCH \
            --mode eval \
            --layer $LAYER
    done
done

# ── Q̂² by A-2 preference (key v2 result) ─────────────────────────────────────
echo ""
echo "=== v2 Q̂² by A-2 preference (VE value inference test) ==="
for EXP in v2_exp_b_base v2_exp_b_mg_ve; do
    echo ""
    echo "--- $EXP ---"
    python analyze/plot_q2_by_pref.py \
        --result_base data/result/${EXP}/${SEED}/test \
        --epoch $EPOCH \
        --mode eval \
        --prefix v2_fixed \
        --save_dir data/result/${EXP}/${SEED}/q2_by_pref/
done

echo ""
echo "======================================================================"
echo "=== v2 Analysis done at: $(date) ==="
echo "======================================================================"
echo ""
echo "Key result (v2 hypothesis):"
echo "  If Q̂²(Red) > Q̂²(Green) for v2_exp_b_mg_ve:"
echo "  → A-1 uses own value experience (red +1, green -2) to infer A-2 preference"
echo ""
echo "Decision guide:"
echo "  FixedA2 other-layer acc >= 90%  → h² encodes static preference"
echo "  Q̂²(Red) - Q̂²(Green) > 0       → value-based preference inference confirmed"
