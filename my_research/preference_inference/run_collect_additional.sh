#!/bin/bash
# Collect additional validation test datasets.
# Usage: bash run_collect_additional.sh
# Run inside Docker: docker exec kusano_research bash -c "cd /work/my_research/preference_inference && bash run_collect_additional.sh"
set -e

PI=/work/my_research/preference_inference
cd $PI

export MESA_GL_VERSION_OVERRIDE=3.3

DATASETS=(
    test_self_rl_other_random
    test_self_rl_other_fixed_red
    test_self_rl_other_fixed_green
    test_self_rl_other_fixed_blue
    test_self_rl_other_fixed_yellow
)

for DATASET in "${DATASETS[@]}"; do
    echo "=== Collecting $DATASET ==="
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
        python simulation/collect_data.py --dataset $DATASET
    echo "Done: $DATASET"
done

echo "=== All additional data collection done ==="
