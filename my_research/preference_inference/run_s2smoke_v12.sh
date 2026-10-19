#!/bin/bash
# v12 checks on GPU2 (docs/v12_experiment_log.md Sec.21-22): (1) G5 with the B1 secondary read-out on (P-c) ep150;
# (2) stage-2 dry run (v12_test_s2_*: 2 epochs from (P-c) ep150, not a result) and the stage-2 evaluation /
# figure scripts on it. Run INSIDE the container from the v11 worktree's my_research/preference_inference.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=2
L=data/result/v12/trial/logs/s2smoke; mkdir -p $L
python -u analyze/v12/eval_g5_v12.py --exp_fmt v12_trial_pc_{}_mse --sub trial/g5_pc_bg > $L/g5_bg.log 2>&1; echo "g5_bg exit $?" >> $L/done
for st in green pref; do for c in random mixed; do
  exp=v12_test_s2_${st}_${c}_s0; mkdir -p data/result/$exp
  python -u train_v12.py --exp_config $exp --seed 0 --device cuda:0 --cudnn_deterministic --data_load_memory > data/result/$exp/train_seed0.log 2>&1 &
done; done
wait; echo "train exit $?" >> $L/done
R="green:random:0 green:mixed:0 pref:random:0 pref:mixed:0"
for w in eval_s2 t1t3 t4t6 m1d_other; do
  python -u analyze/v12/s2run.py $w --s2prefix v12_test_s2 --out trial/s2smoke --runs $R --epochs 2 > $L/$w.log 2>&1; echo "$w exit $?" >> $L/done
done
python -u analyze/v12/figs_s2extra_v12.py --exp_fmt v12_test_s2_green_{}_s0 --seed 0 --epoch 2 --tag smoke --out trial/s2smoke/figs \
  --t4_per data/result/v12/trial/s2smoke/t4t6/green_{}_s0_ep2_per.npz > $L/figs_extra.log 2>&1; echo "figs_extra exit $?" >> $L/done
