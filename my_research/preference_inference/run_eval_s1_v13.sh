#!/bin/bash
# v13 stage-1 evaluation (docs/v13_experiment_log.md Sec.3.3, 3.4, 7). v12 measures run through
# analyze/v13/v12run.py (no v12 code changed; the no-experience runs are the v12 ones, read only).
# Run INSIDE the container from my_research/preference_inference. GPU via $GPU (7, 1 or 2 only).
#   GPU=k bash run_eval_s1_v13.sh PAIR <s>   # everything per pair (mse late-5, l1 late-5, references)
#   GPU=k bash run_eval_s1_v13.sh AGG        # gates, subsets, comparison, figures, convergence, encoder check
set -u
cd "$(dirname "$0")"
GPU=${GPU:-7}
case $GPU in 1|2|7) ;; *) echo "GPU $GPU is not allowed for v13"; exit 1 ;; esac
export CUDA_VISIBLE_DEVICES=$GPU
L=data/result/v13/logs/eval_s1; mkdir -p $L
R="python -u analyze/v13/v12run.py"
EPS="160 170 180 190 200"
st=$1; shift
case $st in
PAIR)
  s=$1
  # G1-G3 (mse late-5) and the repeat of ep200 for the bit-identity check
  $R trial --exp_fmt v13_s1_{}_mse --seed $s --epochs $EPS --no_p2 --out ../v13/main/s1eval/s$s > $L/A_s$s.log 2>&1
  $R trial --exp_fmt v13_s1_{}_mse --seed $s --epochs 200 --no_p2 --tag _rep --out ../v13/main/s1eval/s$s > $L/A_rep_s$s.log 2>&1
  # selection criterion (h2 -> A-2 position, mse ep200)
  $R h2 --runs random:$s:200 mixed:$s:200 --sub ../v13/main/h2_check_s$s > $L/h2_s$s.log 2>&1
  # G5a/G5b (l1 late-5) with per-episode values; D6-like secondary at ep200
  for e in $EPS; do
    $R g5 --exp_fmt v13_s1_{}_l1_s$s --seed $s --epoch $e --sub ../v13/main/g5/s$s > $L/G5_s${s}_e$e.log 2>&1
  done
  $R d6 --exp_fmt v13_s1_{}_l1_s$s --seed $s --epoch 200 --sub ../v13/main/d6/s$s > $L/D6_s$s.log 2>&1
  # descriptive: tracking split by near red / away, on the common, 50% and 80% tests (Sec.3.3, Sec.7)
  $R track --exp_fmt v13_s1_{}_mse --seed $s --out v13/main/track/s$s > $L/track_s$s.log 2>&1
  # v12 reference (50% experienced) for the subsets and the v13 - v12 comparison (Sec.3.4)
  for e in $EPS; do
    $R g5 --exp_fmt v13ref_s1_{}_l1_s$s --seed $s --epoch $e --conds mixed --sub ../v13/v12ref/g5/s$s > $L/G5ref_s${s}_e$e.log 2>&1
  done
  $R track --exp_fmt v13ref_s1_{}_mse --seed $s --conds mixed --out v13/v12ref/track/s$s > $L/trackref_s$s.log 2>&1
  # PCA and prediction images (as v12)
  $R pca --exp_fmt v13_s1_{}_mse --seed $s --epoch 200 --sub ../v13/main/pca > $L/pca_s$s.log 2>&1
  $R pred_images --exp_fmt v13_s1_{}_l1_s$s --epoch 200 --ae_fmt v12_fig5_s0 --seed $s --data common --sub ../v13/main/pred_images > $L/pred_s$s.log 2>&1
  echo "exit $?" > $L/PAIR_s$s.done ;;
AGG)
  SEEDS=${SEEDS:-"0 1 2"}
  python analyze/v12/aggregate_s1_v12.py --root ../v13/main/s1eval --seeds $SEEDS > $L/agg_s1.log 2>&1
  python analyze/v12/aggregate_g5_v12.py --root ../v13/main/g5 --seeds $SEEDS > $L/agg_g5.log 2>&1
  for s in $SEEDS; do python analyze/v12/figs_g5_v12.py --sub ../v13/main/g5/s$s --epoch 200 >> $L/figs_g5.log 2>&1; done
  python analyze/v13/subsets_v13.py --seeds $SEEDS > $L/subsets.log 2>&1
  $R convergence --stage mse --exp_fmt v13_s1_{c}_mse --seeds $SEEDS --sub ../v13/main/convergence > $L/conv_mse.log 2>&1
  $R convergence --stage l1 --exp_fmt v13_s1_{c}_l1_s{s} --seeds $SEEDS --sub ../v13/main/convergence > $L/conv_l1.log 2>&1
  python - $SEEDS > $L/encoder_check.log 2>&1 <<'PY'
import sys, torch
ref = torch.load('/work/data/result/exp1_l1_1000/0/model/00200.pth', map_location='cpu')['model']
ks = [k for k in ref if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module', 'share_lns'))]
for s in map(int, sys.argv[1:]):
    for exp in ('v13_s1_mixed_mse', f'v13_s1_mixed_l1_s{s}'):
        sd = torch.load(f'data/result/{exp}/{s}/model/00200.pth', map_location='cpu')['model']
        print(exp, s, 200, 'encoders equal to v4:', all(torch.equal(sd[k], ref[k]) for k in ks), len(ks))
PY
  echo "exit $?" > $L/AGG.done ;;
*) echo "unknown stage $st"; exit 1 ;;
esac
