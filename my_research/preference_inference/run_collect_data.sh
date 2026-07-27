#!/bin/bash
# Run inside Docker container (kusano_research)
# Usage: bash run_collect_data.sh
# All paths are relative to /work inside the container.

set -e
WORK=/work
PI=$WORK/my_research/preference_inference
cd $PI

XVFB="xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24'"
export MESA_GL_VERSION_OVERRIDE=3.3

echo "=== Collecting new_self_rl_other_cycler (train + test) ==="
$XVFB python simulation/collect_data.py --dataset new_self_rl_other_cycler

echo "=== Collecting new_self_rl_other_red ==="
$XVFB python simulation/collect_data.py --dataset new_self_rl_other_red

echo "=== Collecting new_self_rl_other_green ==="
$XVFB python simulation/collect_data.py --dataset new_self_rl_other_green

echo "=== Collecting new_self_rl_other_blue ==="
$XVFB python simulation/collect_data.py --dataset new_self_rl_other_blue

echo "=== Collecting new_self_rl_other_yellow ==="
$XVFB python simulation/collect_data.py --dataset new_self_rl_other_yellow

echo "=== Collecting new_self_rl_other_stay ==="
$XVFB python simulation/collect_data.py --dataset new_self_rl_other_stay

echo "=== Collecting grid Q-map data ==="
$XVFB python simulation/collect_data.py --dataset new_grid_q_map

echo "=== Done collecting all datasets ==="
