#!/bin/bash
set -u; cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES; export DEVICE=cuda:5
until [ -f data/result/v5_r4_move/0/model/00400.pth ] && ! (for p in /proc/[0-9]*; do [ -r "$p/cmdline" ] && tr "\0" " " < "$p/cmdline" | grep -q "exp_config v5_r4_move" && exit 0; done; exit 1); do sleep 120; done
echo "[p2a-eval] $(date -u) training done, running late-5 eval"
python analyze/check_convergence.py v5_r4_move --seeds 0 2>&1 | grep -v "^saved:"
bash analyze/run_eval_lateckpt.sh v5_r4_move 400 0 r2_a1random_a2rl > logs/p2a_late5.log 2>&1
echo "[p2a-eval] late-5 harness rc=$?"
for ep in 360 370 380 390 400; do
  python analyze/r4_ve_eval.py --exp_config v5_r4_move --epoch $ep --seed 0 --eval_seed 0 \
    --a2_critic_path data/model/v3_rl_a2_critic.pth \
    --data_h5 data/data/r2_a1random_a2rl/data.h5 --target_pos=-9,-9 \
    --label v5_r4_move_ep${ep} 2>&1 | grep -vE "it/s\]|it\]" | grep -E "real |zero |true_q2 |cos_sim|A-1 self|A-2 other|Saved:"
done
echo "[p2a-eval] DONE $(date -u)  -- saved.h5 cleanup pending manual review of JSONs"
