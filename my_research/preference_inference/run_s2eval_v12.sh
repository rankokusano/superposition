#!/bin/bash
# v12 stage-2 evaluation of one pair (docs/v12_experiment_log.md Sec.28, 31). Waits for the pair's 4 runs, checks
# the outputs, then evaluates in the order requested (Kusano-san, 2026-10-20):
#   1 primary T2/T5 (+ corr(Q2, P_A2), Q2 -> A-1 / A-2 step direction; same script) and a repeat for bit-identity,
#   2 priority secondary T3 / Q2 spread / T1, 3 D6 (data-only reference, once), 4 remaining secondary + figures,
#   5 interim summary and interim primary aggregation for this pair.
# Usage INSIDE the container (v11 worktree's my_research/preference_inference), detached:
#   setsid nohup bash run_s2eval_v12.sh <seed> <gpu> > /dev/null 2>&1 < /dev/null &
# Env (for the dry run): PREFIX=v12_s2 OUT=stage2 EPOCHS="160 170 180 190 200" WAIT=1
set -u
cd "$(dirname "$0")"
s=$1; gpu=$2
[ "$gpu" = 0 ] && { echo "GPU0 is not allowed"; exit 1; }
PREFIX=${PREFIX:-v12_s2}; OUT=${OUT:-stage2}; EPOCHS=${EPOCHS:-"160 170 180 190 200"}; WAIT=${WAIT:-1}
export CUDA_VISIBLE_DEVICES=$gpu
L=data/result/v12/$OUT/logs/s$s; mkdir -p $L
R="green:random:$s green:mixed:$s pref:random:$s pref:mixed:$s"
last=${EPOCHS##* }
if [ "$WAIT" = 1 ]; then
  for st in green pref; do for c in random mixed; do
    until [ -f data/result/${PREFIX}_${st}_${c}_s$s/done_seed$s ]; do sleep 120; done
  done; done
fi
{
  echo "start $(date)"
  for st in green pref; do for c in random mixed; do d=data/result/${PREFIX}_${st}_${c}_s$s
    echo "$st $c: done=$(cat $d/done_seed$s 2>/dev/null) ckpts=$(ls $d/$s/model | wc -l) tracebacks=$(grep -c Traceback $d/train_seed$s.log)"
  done; done
} > $L/completion.log
for st in green pref; do
  python analyze/v12/v11fig.py convergence --exp_fmt ${PREFIX}_${st}_{c}_s{s} --seeds $s --sub ../v12/$OUT/convergence_${st}_s$s > $L/conv_$st.log 2>&1
done
step() { echo "$1 exit $2 $(date)" >> $L/steps.log; }
# 1 primary (+ corr, step-direction read-outs)
python -u analyze/v12/s2run.py eval_s2 --s2prefix $PREFIX --out $OUT --runs $R --epochs $EPOCHS > $L/eval_s2.log 2>&1; step eval_s2 $?
E=data/result/v12/$OUT/eval; mkdir -p $E/rep_s$s
cp $E/green_mixed_s${s}_ep${last}.json $E/green_mixed_s${s}_ep${last}_per.npz $E/rep_s$s/
python -u analyze/v12/s2run.py eval_s2 --s2prefix $PREFIX --out $OUT --runs green:mixed:$s --epochs $last > $L/eval_s2_rep.log 2>&1; step eval_s2_rep $?
python - $E $s $last >> $L/steps.log 2>&1 <<'PY'
import json, sys, numpy as np
E, s, e = sys.argv[1:]
a, b = json.load(open(f'{E}/rep_s{s}/green_mixed_s{s}_ep{e}.json')), json.load(open(f'{E}/green_mixed_s{s}_ep{e}.json'))
x, y = np.load(f'{E}/rep_s{s}/green_mixed_s{s}_ep{e}_per.npz'), np.load(f'{E}/green_mixed_s{s}_ep{e}_per.npz')
print('bit-identical', a == b and all((x[k] == y[k]).all() for k in x))
PY
# 2 priority secondary: T3, Q2 spread, T1
python -u analyze/v12/s2run.py t1t3 --s2prefix $PREFIX --out $OUT --runs $R --epochs $EPOCHS > $L/t1t3.log 2>&1; step t1t3 $?
# 3 D6 (data-only reference; once per output root)
exec 9> data/result/v12/$OUT/.d6.lock; flock 9
[ -f $E/d6_PA2.json ] || { python -u analyze/v12/s2run.py eval_s2 --s2prefix $PREFIX --out $OUT --runs d6:data:0 > $L/d6_eval.log 2>&1; step d6_eval $?; }
[ -f data/result/v12/$OUT/t1t3/d6_PA2.json ] || { python -u analyze/v12/s2run.py t1t3 --s2prefix $PREFIX --out $OUT --runs d6:data:0 > $L/d6_t1t3.log 2>&1; step d6_t1t3 $?; }
flock -u 9
# 4 remaining secondary and figures
python -u analyze/v12/s2run.py t4t6 --s2prefix $PREFIX --out $OUT --runs $R --epochs $EPOCHS > $L/t4t6.log 2>&1; step t4t6 $?
python -u analyze/v12/s2run.py m1d_other --s2prefix $PREFIX --out $OUT --runs $R --epochs $EPOCHS > $L/m1d_other.log 2>&1; step m1d_other $?
for st in green pref; do
  python -u analyze/v12/figs_s2extra_v12.py --exp_fmt ${PREFIX}_${st}_{}_s$s --seed $s --epoch $last --tag ${st}_s$s --out $OUT/figs \
    --t4_per data/result/v12/$OUT/t4t6/${st}_{}_s${s}_ep${last}_per.npz > $L/figs_$st.log 2>&1; step figs_$st $?
done
# 5 interim summaries
python analyze/v12/s2pair_summary_v12.py --seed $s --out $OUT --epochs $EPOCHS > $L/summary.log 2>&1; step summary $?
python analyze/v12/aggregate_s2_v12.py --seeds $s --out $OUT --epochs $EPOCHS --tag interim_s$s > $L/aggregate_interim.log 2>&1; step aggregate_interim $?
echo "all done $(date)" >> $L/steps.log
