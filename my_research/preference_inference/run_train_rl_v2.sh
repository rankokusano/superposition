#!/bin/bash
# Train the v2 RL agent (red-preference + green-aversion) from scratch.
# Expected training time: ~60-90 min on GPU.
#
# Run inside Docker:
#   docker exec kusano_research bash -c "cd /work/my_research/preference_inference && bash run_train_rl_v2.sh"
set -e

PI=/work/my_research/preference_inference
cd $PI

export MESA_GL_VERSION_OVERRIDE=3.3

echo "=== Training v2 RL agent (red +1, green -2) ==="
echo "Output: data/model/v2_rl_actor.pth, data/model/v2_rl_critic.pth"
echo "Started at: $(date)"

xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
    python simulation/train_rl_v2.py 2>&1 | tee /tmp/train_rl_v2.log

echo ""
echo "=== RL training done at: $(date) ==="
echo "Log saved to /tmp/train_rl_v2.log"
