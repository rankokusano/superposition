#!/bin/bash
# v11 evaluation of the main run (docs/v11_experiment_log.md Sec.7.18). Run INSIDE the
# container from the v11 worktree's my_research/preference_inference:
#   bash run_eval_main_v11.sh A   # after the mse stage (main criteria, mse late-5)
#   bash run_eval_main_v11.sh B   # after the l1 stage (secondary values on l1 late-5)
#   bash run_eval_main_v11.sh C   # Fig.5 decoders (train), Fig.5 maps/numbers, prediction images
# GPU via $GPU (default 7). Never GPU0.
set -u
cd "$(dirname "$0")"
GPU=${GPU:-7}
[ "$GPU" = 0 ] && { echo "GPU0 is not allowed"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU
L=data/result/v11/logs/eval_main
mkdir -p $L
case $1 in
A)
  python analyze/v11/check_convergence_v11.py --stage mse --sub main/convergence > $L/conv_mse.log 2>&1
  for s in 0 1 2; do for e in 160 170 180 190 200; do
    python analyze/v11/eval_m1_v11.py --exp_fmt v11_s1_{}_mse --seed $s --epoch $e --n_train 3000 --sub main/m1 > $L/m1_s${s}_e${e}.log 2>&1
    python analyze/v11/m1d_masked_v11.py --exp_fmt v11_s1_{}_mse --seed $s --epoch $e --sub main/m1 > $L/m1d_s${s}_e${e}.log 2>&1
  done
    python analyze/v11/m5m6_v11.py --exp_fmt v11_s1_{}_mse --seed $s --epoch 200 --sub main/m5m6 > $L/m5m6_s${s}.log 2>&1
    python analyze/v11/pca_v11.py --exp_fmt v11_s1_{}_mse --seed $s --epoch 200 --sub main/pca > $L/pca_s${s}.log 2>&1
  done
  python analyze/v11/aggregate_main_v11.py --m1_fmt "main/m1/m1_v11_s1_X_mse_s{seed}_ep{epoch}_pmaskoff.json" \
      --m1d_fmt "main/m1/m1d_masked_v11_s1_X_mse_s{seed}_ep{epoch}.json" --convergence main/convergence/convergence_mse.json \
      --sub main/gate --label mse_late5 > $L/gate_mse.log 2>&1
  ;;
B)
  python analyze/v11/check_convergence_v11.py --stage l1 --sub main/convergence > $L/conv_l1.log 2>&1
  for s in 0 1 2; do for e in 160 170 180 190 200; do
    python analyze/v11/eval_m1_v11.py --exp_fmt v11_s1_{}_l1_s$s --seed $s --epoch $e --n_train 3000 --sub main/m1_l1 > $L/m1l1_s${s}_e${e}.log 2>&1
    python analyze/v11/m1d_masked_v11.py --exp_fmt v11_s1_{}_l1_s$s --seed $s --epoch $e --sub main/m1_l1 > $L/m1dl1_s${s}_e${e}.log 2>&1
  done; done
  python analyze/v11/aggregate_main_v11.py --m1_fmt "main/m1_l1/m1_v11_s1_X_l1_s{seed}_s{seed}_ep{epoch}_pmaskoff.json" \
      --m1d_fmt "main/m1_l1/m1d_masked_v11_s1_X_l1_s{seed}_s{seed}_ep{epoch}.json" --convergence main/convergence/convergence_l1.json \
      --sub main/gate --label l1_late5_secondary > $L/gate_l1.log 2>&1
  ;;
C)
  for c in random mixed; do for s in 0 1 2; do
    test -f data/result/v11_s1_${c}_l1_s${s}/${s}/model/00200.pth || { echo "missing l1 ep200 for $c s$s"; exit 1; }
    python -u train_v11.py --exp_config v11_fig5_${c}_s${s} --seed $s --device cuda:0 --cudnn_deterministic > $L/fig5train_${c}_s${s}.log 2>&1
    python analyze/v11/fig5_v11.py --ae v11_fig5_${c}_s${s} --seed $s --epoch 100 --sub main/fig5 > $L/fig5_${c}_s${s}.log 2>&1
  done; done
  for s in 0 1 2; do
    for d in common own; do
      python analyze/v11/pred_images_v11.py --exp_fmt v11_s1_{}_l1_s$s --seed $s --epoch 200 --ae_fmt v11_fig5_{}_s$s --ae_epoch 100 \
          --data $d --sub main/pred_images > $L/pred_${d}_s${s}.log 2>&1
    done
  done
  ;;
*) echo "usage: $0 A|B|C"; exit 1;;
esac
echo "stage $1 done"
