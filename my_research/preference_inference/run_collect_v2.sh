#!/bin/bash
# Collect all v2 datasets using the v2 RL agent (must be trained first via run_train_rl_v2.sh).
#
# Run inside Docker:
#   docker exec kusano_research bash -c "cd /work/my_research/preference_inference && bash run_collect_v2.sh"
set -e

PI=/work/my_research/preference_inference
cd $PI

export MESA_GL_VERSION_OVERRIDE=3.3

DATASETS=(
    v2_self_rl_other_stay
    v2_self_rl_other_random_landmark
    v2_fixed_red
    v2_fixed_green
    v2_fixed_blue
    v2_fixed_yellow
    v2_grid_q_map
)

echo "=== v2 data collection started at: $(date) ==="

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "--- Collecting: $DATASET ---"
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data.py --dataset $DATASET
    echo "Done: $DATASET"
done

echo ""
echo "=== v2 data collection done at: $(date) ==="
