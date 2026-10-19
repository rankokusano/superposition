#!/bin/bash
# v12 stage-2 dry run, evaluation part only (re-run after fixing load_v12 for exp3 configs; docs/v12_experiment_log.md Sec.21). GPU2.
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=2
L=data/result/v12/trial/logs/s2smoke; mkdir -p $L
R="green:random:0 green:mixed:0 pref:random:0 pref:mixed:0"
for w in eval_s2 t1t3 t4t6 m1d_other; do
  python -u analyze/v12/s2run.py $w --s2prefix v12_test_s2 --out trial/s2smoke --runs $R --epochs 2 > $L/$w.log 2>&1; echo "$w exit $?" >> $L/done_eval
done
python -u analyze/v12/figs_s2extra_v12.py --exp_fmt v12_test_s2_green_{}_s0 --seed 0 --epoch 2 --tag smoke --out trial/s2smoke/figs \
  --t4_per data/result/v12/trial/s2smoke/t4t6/green_{}_s0_ep2_per.npz > $L/figs_extra.log 2>&1; echo "figs_extra exit $?" >> $L/done_eval
