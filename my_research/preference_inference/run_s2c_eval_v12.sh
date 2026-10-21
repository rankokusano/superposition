#!/bin/bash
# v12 stage 2c evaluation of one pair (docs/v12_experiment_log.md Sec.46). Waits for the pair's 4 runs, checks them,
# then: 1 MAIN primary T2/T5 on the hidden test (+ the visible-at-t50 / both-visible reference, corr, sign agreement,
# T3-hidden; same script) and a repeat for bit-identity, 2 priority secondary B3 on the hidden test, 3 D6 (once),
# 4 reference: the registered visible test (v11 sets, s2run eval_s2 and t1t3), 5 interim aggregations.
# Usage INSIDE the container, detached: setsid nohup bash run_s2c_eval_v12.sh <seed> <gpu> > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
s=$1; gpu=$2
[ "$gpu" = 0 ] && { echo "GPU0 is not allowed"; exit 1; }
PREFIX=v12_s2c; OUT=stage2c; EPOCHS="160 170 180 190 200"; last=200
export CUDA_VISIBLE_DEVICES=$gpu
L=data/result/v12/$OUT/logs/s$s; mkdir -p $L
R="green:random:$s green:mixed:$s pref:random:$s pref:mixed:$s"
for st in green pref; do for c in random mixed; do
  until [ -f data/result/${PREFIX}_${st}_${c}_s$s/done_seed$s ]; do sleep 120; done
done; done
{
  echo "start $(date)"
  for st in green pref; do for c in random mixed; do d=data/result/${PREFIX}_${st}_${c}_s$s
    echo "$st $c: done=$(cat $d/done_seed$s 2>/dev/null) ckpts=$(ls $d/$s/model | wc -l) tracebacks=$(grep -c Traceback $d/train_seed$s.log) input=$(grep -c 'input view from data key self_vision_in' $d/train_seed$s.log)"
  done; done
} > $L/completion.log
for st in green pref; do
  python analyze/v12/v11fig.py convergence --exp_fmt ${PREFIX}_${st}_{c}_s{s} --seeds $s --sub ../v12/$OUT/convergence_${st}_s$s > $L/conv_$st.log 2>&1
done
step() { echo "$1 exit $2 $(date)" >> $L/steps.log; }
# 1 main primary on the hidden test (+ reference subsets and secondary in the same pass), repeat for bit-identity
python -u analyze/v12/eval_s2c_v12.py --prefix $PREFIX --out $OUT --runs $R --epochs $EPOCHS > $L/eval_s2c.log 2>&1; step eval_s2c $?
E=data/result/v12/$OUT/eval; mkdir -p $E/rep_s$s
cp $E/green_mixed_s${s}_ep${last}.json $E/green_mixed_s${s}_ep${last}_per.npz $E/rep_s$s/
python -u analyze/v12/eval_s2c_v12.py --prefix $PREFIX --out $OUT --runs green:mixed:$s --epochs $last > $L/eval_s2c_rep.log 2>&1; step eval_s2c_rep $?
python - $E $s $last >> $L/steps.log 2>&1 <<'PY'
import json, sys, numpy as np
E, s, e = sys.argv[1:]
a, b = json.load(open(f'{E}/rep_s{s}/green_mixed_s{s}_ep{e}.json')), json.load(open(f'{E}/green_mixed_s{s}_ep{e}.json'))
x, y = np.load(f'{E}/rep_s{s}/green_mixed_s{s}_ep{e}_per.npz'), np.load(f'{E}/green_mixed_s{s}_ep{e}_per.npz')
print('bit-identical', a == b and all((x[k] == y[k]).all() for k in x))
PY
# 2 priority secondary B3 on the hidden test
for st in green pref; do
  python -u analyze/v12/eval_s2c_b3_v12.py --prefix $PREFIX --out $OUT --setting $st --seed $s --epochs $EPOCHS > $L/b3_$st.log 2>&1; step b3_$st $?
done
# 3 D6 on the hidden-test subsets (data only; once)
exec 9> data/result/v12/$OUT/.d6.lock; flock 9
[ -f $E/d6_PA2.json ] || { python -u analyze/v12/eval_s2c_v12.py --out $OUT --runs d6:data:0 > $L/d6.log 2>&1; step d6 $?; }
flock -u 9
# 4 reference: the registered visible test (same scripts as stage 2 / 2b; input = original view)
python -u analyze/v12/s2run.py eval_s2 --s2prefix $PREFIX --out ${OUT}_vis --runs $R --epochs $EPOCHS > $L/eval_vis.log 2>&1; step eval_vis $?
exec 8> data/result/v12/$OUT/.d6vis.lock; flock 8
[ -f data/result/v12/${OUT}_vis/eval/d6_PA2.json ] || { python -u analyze/v12/s2run.py eval_s2 --s2prefix $PREFIX --out ${OUT}_vis --runs d6:data:0 > $L/d6_vis.log 2>&1; step d6_vis $?; }
flock -u 8
python -u analyze/v12/s2run.py t1t3 --s2prefix $PREFIX --out ${OUT}_vis --runs $R --epochs $EPOCHS > $L/t1t3_vis.log 2>&1; step t1t3_vis $?
# 5 interim aggregations (main = hidden; references = visible-at-t50 subset of the hidden test, visible test)
for o in $OUT ${OUT}_hidvis ${OUT}_vis; do
  python analyze/v12/aggregate_s2_v12.py --seeds $s --out $o --epochs $EPOCHS --tag interim_s$s > $L/aggregate_interim_$o.log 2>&1; step aggregate_$o $?
done
echo "all done $(date)" >> $L/steps.log
