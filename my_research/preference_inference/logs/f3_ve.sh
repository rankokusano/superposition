#!/bin/bash
set -u; cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES; export DEVICE=cuda:6
gen() { # exp epoch dataset
  python test.py --exp_config "$1" --seed 0 --device cuda:6 --test_epoch "$2" \
    --test_data_name "$3" --test_batch_size 10 --test_modes eval \
    --save_targets state self_position other_position --test_name "${3}_f3ve" \
    --eval_seed 0 --cudnn_deterministic > "logs/f3_${1}.log" 2>&1
  echo "[f3] $1 rc=$?"
}
gen r4_move_exp3parity 400 r2_a1random_a2rl
gen r4_move_sym 400 r2_a1red_a2red
gen v5_r4_move 400 r2_a1random_a2rl
echo "[f3] evals done, running plot_state + PC-scan"
for spec in "r4_move_exp3parity r2_a1random_a2rl 400 P0-4" "r4_move_sym r2_a1red_a2red 400 P1.75" "v5_r4_move r2_a1random_a2rl 400 P2-a"; do
  set -- $spec
  d=data/result/$1/0/test/$2_f3ve/save
  find "$d" -name pca.pkl -delete 2>/dev/null
  python analyze/plot_state.py --result_dir "$d" --epoch $3 --mode eval \
    --plot other_position --plot_layer other --analyze_layer other --dim_x 0 --dim_y 1 --pca_components 4 2>&1 | grep Saved
done
echo "[f3] DONE"
