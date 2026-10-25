#!/bin/bash
# v15 Sec.20.4: no-experience stage-1 data (plan C'), train 3000 + test 300 + _1000t, key v15_s1. CPU, nice 10, 8 at a time.
# Two generations of train 0..19 must be bit-identical (merge_chunks_v15.py also refuses empty views), then the
# non-emptiness check of the result. Run INSIDE the container, detached.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= MESA_GL_VERSION_OVERRIDE=3.3
K=data/result/v15/datacheck; N=v15_s1_goalcap_half_3000; mkdir -p $K logs/v15
run_one() {
  set -- $1   # split start tag n
  if [ "$3" = chunk ]; then out=data/data/v15_s1_goalcap_half_3000/chunks/$1_$2.h5; else out=data/result/v15/datacheck/v15_s1_goalcap_half_3000_$3.h5; fi
  mkdir -p $(dirname $out)
  nice -n 10 xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' python simulation/collect_data_v15.py \
    --dataset v15_s1_goalcap_half --split $1 --start $2 --n $4 --save_path $out > logs/v15/collect_goalcap_$1_$2_$3.log 2>&1
  rc=$?; echo "$(date +%T) done $* (exit $rc)"
}
export -f run_one
{ for s in 0 500 1000 1500 2000 2500; do echo "train $s chunk 500"; done
  echo "test 0 chunk 300"; echo "train 0 check_a 20"; echo "train 0 check_b 20"; } | xargs -P 8 -I{} bash -c 'run_one "{}"'
if grep "done" logs/v15_goalcap_driver.log | grep -vq "exit 0"; then echo "chunk job failed"; exit 1; fi
nice -n 10 python analyze/v15/merge_chunks_v15.py --delete_chunks --first1000 --name $N \
  --parts $(for s in 0 500 1000 1500 2000 2500; do echo -n "train:$s:500:chunk "; done) test:0:300:chunk \
  --check $K/${N}_check_a.h5,$K/${N}_check_b.h5,train:0 > logs/v15/merge_goalcap.log 2>&1
rc=$?; echo "merge exit $rc"; [ $rc = 0 ] || exit 1
nice -n 10 python analyze/v15/check_nonempty_v15.py $N v15_s1_goalcap_half_1000t > logs/v15/nonempty_goalcap.log 2>&1
rc=$?; echo "nonempty exit $rc"
echo ALL_DONE
