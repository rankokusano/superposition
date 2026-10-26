#!/bin/bash
# v15 Sec.28 (B): stage-1 experienced data with A-1's rank = Y (Green > Cyan > Blue > Red), trial scale (train 1000 + test 300),
# key v15_s1. Two generations of train 0..19 must be bit-identical; non-emptiness check. CPU, nice 10. Run INSIDE the container.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= MESA_GL_VERSION_OVERRIDE=3.3
K=data/result/v15/datacheck; N=v15_s1_rankY_p80_half_1000t; mkdir -p $K logs/v15
run_one() {
  set -- $1
  if [ "$3" = chunk ]; then out=data/data/v15_s1_rankY_p80_half_1000t/chunks/$1_$2.h5; else out=data/result/v15/datacheck/v15_s1_rankY_p80_half_1000t_$3.h5; fi
  mkdir -p $(dirname $out)
  nice -n 10 xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' python simulation/collect_data_v15.py \
    --dataset v15_s1_rankY_p80_half --split $1 --start $2 --n $4 --save_path $out > logs/v15/collect_rankY_$1_$2_$3.log 2>&1
  rc=$?; echo "$(date +%T) done $* (exit $rc)"
}
export -f run_one
printf '%s\n' "train 0 chunk 500" "train 500 chunk 500" "test 0 chunk 300" "train 0 check_a 20" "train 0 check_b 20" | xargs -P 5 -I{} bash -c 'run_one "{}"'
if grep "done" logs/v15_rankY_driver.log | grep -vq "exit 0"; then echo "chunk job failed"; exit 1; fi
nice -n 10 python analyze/v15/merge_chunks_v15.py --delete_chunks --name $N --parts train:0:500:chunk train:500:500:chunk test:0:300:chunk \
  --check $K/${N}_check_a.h5,$K/${N}_check_b.h5,train:0 > logs/v15/merge_rankY.log 2>&1
rc=$?; echo "merge exit $rc"; [ $rc = 0 ] || exit 1
nice -n 10 python analyze/v15/check_nonempty_v15.py $N > logs/v15/nonempty_rankY.log 2>&1; echo "nonempty exit $?"
echo ALL_DONE
