#!/bin/bash
# v7 closing checks (docs/v7_experiment_log.md §2.14). Run inside Docker.
PI=/work/my_research/preference_inference
C=data/result/v7_irl/retrain/closing
F="python -u analyze/feasibility_goals_v7.py"
run() {  # gpu log args...
  local GPU=$1 LOG=$2; shift 2
  (cd $PI && CUDA_VISIBLE_DEVICES=$GPU xvfb-run --auto-servernum $F "$@" > logs/$LOG 2>&1; echo "$LOG EXIT=$?")
}
( run 1 v7_closing_fixed_crit_seed3.log --rollout --policy fixed --fixed_dirs data/result/v7_irl/retrain/seed3_condA_r1/check2_maps.npz --rollout_key v7_retrain_seed3_condA_r1 --out_dir $C/fixed_crit_seed3
  run 1 v7_closing_fixed_corner_seed3key.log --rollout --policy fixed --fixed_dirs corner --rollout_key v7_retrain_seed3_condA_r1 --out_dir $C/fixed_corner_seed3key ) &
( run 2 v7_closing_fixed_crit_v6.log --rollout --policy fixed --fixed_dirs data/result/v7_irl/retrain/v6_seed2_ref/check2_maps.npz --rollout_key v7_feasibility --out_dir $C/fixed_crit_v6
  run 2 v7_closing_fixed_corner_v6key.log --rollout --policy fixed --fixed_dirs corner --rollout_key v7_feasibility --out_dir $C/fixed_corner_v6key ) &
( run 3 v7_closing_hidreset_seed1.log --rollout --check1 --actor_hidden_reset --actor_path data/model/v7/v7_rl_actor_seed1_condA.pth --rollout_key v7_retrain_seed1_condA --out_dir $C/hidreset_seed1
  run 3 v7_closing_hidreset_seed3.log --rollout --check1 --actor_hidden_reset --actor_path data/model/v7/v7_rl_actor_seed3_condA_r1.pth --rollout_key v7_retrain_seed3_condA_r1 --out_dir $C/hidreset_seed3 ) &
wait
cd $PI && python analyze/closing_checks_v7.py > logs/v7_closing_checks.log 2>&1; echo "closing_checks EXIT=$?"
echo "CLOSING CHECKS DONE"
