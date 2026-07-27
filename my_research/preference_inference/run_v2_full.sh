#!/bin/bash
# Full v2 experiment pipeline: RL training → data → model training → test → analysis.
#
# Run inside Docker:
#   docker exec -it kusano_research bash
#   cd /work/my_research/preference_inference
#   bash run_v2_full.sh
set -e

PI=/work/my_research/preference_inference
cd $PI

SEED=${1:-0}

echo "======================================================================"
echo "=== v2 Full Pipeline ==="
echo "======================================================================"
echo "Started at: $(date)"
echo ""

# Step 1: Train v2 RL agent
echo "--- Step 1: RL training ---"
bash run_train_rl_v2.sh

# Step 2: Collect v2 datasets
echo ""
echo "--- Step 2: Data collection ---"
bash run_collect_v2.sh

# Step 3: Train v2 Superposition models
echo ""
echo "--- Step 3: Superposition model training ---"
bash run_training_v2.sh $SEED

# Step 4: Generate saved.h5 for evaluation datasets
echo ""
echo "--- Step 4: Test (generate saved.h5) ---"
bash run_test_v2.sh 200 $SEED

# Step 5: Analysis
echo ""
echo "--- Step 5: Analysis ---"
bash run_analysis_v2.sh 200 $SEED

echo ""
echo "======================================================================"
echo "=== v2 Full Pipeline Done at: $(date) ==="
echo "======================================================================"
