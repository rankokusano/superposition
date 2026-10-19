#!/bin/bash
# v12 stage-1 evaluation of the main run (docs/v12_experiment_log.md Sec.11, 13, 16). Run INSIDE the
# container from the v11 worktree's my_research/preference_inference. GPU via $GPU (never 0).
#   GPU=k bash run_eval_s1_v12.sh A <seed>     # mse late-5: tracking, M1 (hidden/visible), colour, G3, alignment
#   GPU=k bash run_eval_s1_v12.sh G5 <seed>    # l1 late-5: G5a/G5b; ep200: G5 secondary (D6-like)
#   GPU=k bash run_eval_s1_v12.sh A_rep <seed> # repeat ep200 of A for the bit-identity check
#   GPU=k bash run_eval_s1_v12.sh FIG5         # Fig.5 Decoder (once) + read-out + encoder check of all runs
#   GPU=k bash run_eval_s1_v12.sh AGG          # gates, figures, convergence, PCA, prediction images
set -u
cd "$(dirname "$0")"
GPU=${GPU:-7}
[ "$GPU" = 0 ] && { echo "GPU0 is not allowed"; exit 1; }
export CUDA_VISIBLE_DEVICES=$GPU
L=data/result/v12/logs/eval_s1; mkdir -p $L
st=$1; shift
EPS="160 170 180 190 200"
case $st in
A)
  s=$1
  python -u analyze/v12/eval_trial_v12.py --exp_fmt v12_s1_{}_mse --seed $s --epochs $EPS --no_p2 --out main/s1eval/s$s > $L/A_s$s.log 2>&1 ;;
A_rep)
  s=$1
  python -u analyze/v12/eval_trial_v12.py --exp_fmt v12_s1_{}_mse --seed $s --epochs 200 --no_p2 --tag _rep --out main/s1eval/s$s > $L/A_rep_s$s.log 2>&1 ;;
G5)
  s=$1
  for e in $EPS; do
    python -u analyze/v12/eval_g5_v12.py --exp_fmt v12_s1_{}_l1_s$s --seed $s --epoch $e --sub main/g5/s$s > $L/G5_s${s}_e$e.log 2>&1
  done
  python -u analyze/v12/eval_d6_v12.py --exp_fmt v12_s1_{}_l1_s$s --seed $s --epoch 200 --sub main/d6/s$s > $L/D6_s$s.log 2>&1 ;;
FIG5)
  python -u train_v12.py --exp_config v12_fig5_s0 --seed 0 --device cuda:0 --cudnn_deterministic --data_load_memory > $L/fig5_train.log 2>&1
  python -u analyze/v11/fig5_v11.py --ae v12_fig5_s0 --seed 0 --epoch 100 --sub ../v12/main/fig5 > $L/fig5_eval.log 2>&1
  python - > $L/encoder_check.log 2>&1 <<'PY'
import torch
ref = torch.load('/work/data/result/exp1_l1_1000/0/model/00200.pth', map_location='cpu')['model']
ks = [k for k in ref if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module', 'share_lns'))]
for c in ('random', 'mixed'):
    for s in (0, 1, 2):
        for exp, e in ((f'v12_s1_{c}_mse', 200), (f'v12_s1_{c}_l1_s{s}', 200)):
            sd = torch.load(f'data/result/{exp}/{s}/model/{e:05d}.pth', map_location='cpu')['model']
            print(exp, s, e, 'encoders equal to v4:', all(torch.equal(sd[k], ref[k]) for k in ks))
PY
  ;;
AGG)
  python analyze/v12/aggregate_s1_v12.py --root main/s1eval > $L/agg_s1.log 2>&1
  python analyze/v12/aggregate_g5_v12.py --root main/g5 > $L/agg_g5.log 2>&1
  for s in 0 1 2; do python analyze/v12/figs_g5_v12.py --sub main/g5/s$s --epoch 200 >> $L/figs_g5.log 2>&1; done
  python analyze/v12/v11fig.py convergence --stage mse --exp_fmt v12_s1_{c}_mse --sub ../v12/main/convergence > $L/conv_mse.log 2>&1
  python analyze/v12/v11fig.py convergence --stage l1 --exp_fmt v12_s1_{c}_l1_s{s} --sub ../v12/main/convergence > $L/conv_l1.log 2>&1
  for s in 0 1 2; do
    python analyze/v12/v11fig.py pca --exp_fmt v12_s1_{}_mse --seed $s --epoch 200 --sub ../v12/main/pca > $L/pca_s$s.log 2>&1
    python analyze/v12/v11fig.py pred_images --exp_fmt v12_s1_{}_l1_s$s --epoch 200 --ae_fmt v12_fig5_s0 --seed $s --data common --sub ../v12/main/pred_images > $L/pred_s$s.log 2>&1
  done ;;
*) echo "unknown stage $st"; exit 1 ;;
esac
echo "done $st $*"
