#!/bin/bash
# v11 evaluation of the main run (docs/v11_experiment_log.md Sec.7.18-7.20). Run INSIDE the
# container from the v11 worktree's my_research/preference_inference. GPU via $GPU (never 0).
#   GPU=k bash run_eval_main_v11.sh A <seed> [epochs...]   # mse late-5 per seed: M1 (a',b',c,d), masked M1(d)
#                                                           #   (+ M5/M6, PCA at ep200)
#   bash run_eval_main_v11.sh A_agg                        # convergence (mse), gate judgement, secondary figures
#   GPU=k bash run_eval_main_v11.sh B <seed> [epochs...]   # same M1 metrics on the l1 stage (secondary)
#   bash run_eval_main_v11.sh B_agg                        # convergence (l1), l1 aggregation
#   GPU=k bash run_eval_main_v11.sh C <cond> <seed>        # Fig.5 decoder training + Fig.5 maps/numbers
#   GPU=k bash run_eval_main_v11.sh C_img <seed>           # prediction images (needs both conditions' Fig.5 AE)
set -u
cd "$(dirname "$0")"
GPU=${GPU:-7}
[ "$GPU" = 0 ] && { echo "GPU0 is not allowed"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU
L=data/result/v11/logs/eval_main
mkdir -p $L
st=$1; shift
case $st in
A)
  s=$1; shift; eps=${*:-160 170 180 190 200}
  for e in $eps; do
    python analyze/v11/eval_m1_v11.py --exp_fmt v11_s1_{}_mse --seed $s --epoch $e --n_train 3000 --sub main/m1 > $L/m1_s${s}_e${e}.log 2>&1
    python analyze/v11/m1d_masked_v11.py --exp_fmt v11_s1_{}_mse --seed $s --epoch $e --sub main/m1 > $L/m1d_s${s}_e${e}.log 2>&1
    if [ "$e" = 200 ]; then
      python analyze/v11/m5m6_v11.py --exp_fmt v11_s1_{}_mse --seed $s --epoch 200 --sub main/m5m6 > $L/m5m6_s${s}.log 2>&1
      python analyze/v11/pca_v11.py --exp_fmt v11_s1_{}_mse --seed $s --epoch 200 --sub main/pca > $L/pca_s${s}.log 2>&1
    fi
  done ;;
A_agg)
  python analyze/v11/check_convergence_v11.py --stage mse --sub main/convergence > $L/conv_mse.log 2>&1
  python analyze/v11/aggregate_main_v11.py --m1_fmt "main/m1/m1_v11_s1_X_mse_s{seed}_ep{epoch}_pmaskoff.json" \
      --m1d_fmt "main/m1/m1d_masked_v11_s1_X_mse_s{seed}_ep{epoch}.json" --convergence main/convergence/convergence_mse.json \
      --sub main/gate --label mse_late5 > $L/gate_mse.log 2>&1
  python analyze/v11/figs_secondary_v11.py --m1_fmt "main/m1/m1_v11_s1_X_mse_s{seed}_ep{epoch}_pmaskoff.json" \
      --m5m6_fmt "main/m5m6/m5m6_v11_s1_X_mse_s{seed}_ep{epoch}.json" --epoch 200 --sub main/secondary --label mse_ep200 > $L/secondary.log 2>&1 ;;
B)
  s=$1; shift; eps=${*:-160 170 180 190 200}
  for e in $eps; do
    python analyze/v11/eval_m1_v11.py --exp_fmt v11_s1_{}_l1_s$s --seed $s --epoch $e --n_train 3000 --sub main/m1_l1 > $L/m1l1_s${s}_e${e}.log 2>&1
    python analyze/v11/m1d_masked_v11.py --exp_fmt v11_s1_{}_l1_s$s --seed $s --epoch $e --sub main/m1_l1 > $L/m1dl1_s${s}_e${e}.log 2>&1
  done ;;
B_agg)
  python analyze/v11/check_convergence_v11.py --stage l1 --sub main/convergence > $L/conv_l1.log 2>&1
  python analyze/v11/aggregate_main_v11.py --m1_fmt "main/m1_l1/m1_v11_s1_X_l1_s{seed}_s{seed}_ep{epoch}_pmaskoff.json" \
      --m1d_fmt "main/m1_l1/m1d_masked_v11_s1_X_l1_s{seed}_s{seed}_ep{epoch}.json" --convergence main/convergence/convergence_l1.json \
      --sub main/gate --label l1_late5_secondary > $L/gate_l1.log 2>&1 ;;
C)
  c=$1; s=$2
  test -f data/result/v11_s1_${c}_l1_s${s}/${s}/model/00200.pth || { echo "missing l1 ep200 for $c s$s"; exit 1; }
  python -u train_v11.py --exp_config v11_fig5_${c}_s${s} --seed $s --device cuda:0 --cudnn_deterministic > $L/fig5train_${c}_s${s}.log 2>&1
  python analyze/v11/fig5_v11.py --ae v11_fig5_${c}_s${s} --seed $s --epoch 100 --sub main/fig5 > $L/fig5_${c}_s${s}.log 2>&1 ;;
C_img)
  s=$1
  for d in common own; do
    python analyze/v11/pred_images_v11.py --exp_fmt v11_s1_{}_l1_s$s --seed $s --epoch 200 --ae_fmt v11_fig5_{}_s$s --ae_epoch 100 \
        --data $d --sub main/pred_images > $L/pred_${d}_s${s}.log 2>&1
  done ;;
*) echo "usage: see header"; exit 1;;
esac
echo "stage $st $* done"
