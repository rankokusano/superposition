#!/bin/bash
# v15 Sec.15 data (CPU only, nice 10, 8 at a time). Run INSIDE the container, detached.
#  A v15_s2_c2_p80_half_6000 : phase-2 data (A-1 random, A-2 X/Y, gray half), train 6000 + test 300 (+ _1000t)
#  B v15_s1_random_half_3000 : stage-1 no-experience (A-1 random), train 3000 + test 300 (+ _1000t)
#  C v15_s1_p80_half_6000    : stage-1 experienced, train 0-2999 = v15_s1_p80_half_3000, 3000-5999 new; test = same
#  plus v15_s1_p80_half_1000t (train 0-999 + test of v15_s1_p80_half_3000) for the trial scale.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= MESA_GL_VERSION_OVERRIDE=3.3
K=data/result/v15/datacheck; mkdir -p $K logs/v15
run_one() {
  set -- $1   # dataset name split start n tag
  if [ "$5" = chunk ]; then out=data/data/$1/chunks/$3_$4.h5; else out=data/result/v15/datacheck/$1_$5.h5; fi
  mkdir -p $(dirname $out)
  nice -n 10 xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' python simulation/collect_data_v15.py \
    --dataset $2 --split $3 --start $4 --n $6 --save_path $out > logs/v15/collect_${1}_$3_$4_$5.log 2>&1
  echo "$(date +%T) done $* (exit $?)"
}
export -f run_one
{
  for s in 0 750 1500 2250 3000 3750 4500 5250; do echo "v15_s2_c2_p80_half_6000 v15_s2_c2_p80_half train $s chunk 750"; done
  echo "v15_s2_c2_p80_half_6000 v15_s2_c2_p80_half test 0 chunk 300"
  for r in a b; do echo "v15_s2_c2_p80_half_6000 v15_s2_c2_p80_half train 0 check_$r 20"; done
  for s in 0 750 1500 2250; do echo "v15_s1_random_half_3000 v15_s1_random_half train $s chunk 750"; done
  echo "v15_s1_random_half_3000 v15_s1_random_half test 0 chunk 300"
  for r in a b; do echo "v15_s1_random_half_3000 v15_s1_random_half train 0 check_$r 20"; done
  for s in 3000 3750 4500 5250; do echo "v15_s1_p80_half_6000 v15_s1_p80_half train $s chunk 750"; done
  for r in a b; do echo "v15_s1_p80_half_6000 v15_s1_p80_half train 3000 check_$r 20"; done
} | xargs -P 8 -I{} bash -c 'run_one "{}"'
if grep "done" logs/v15_s2data_driver.log | grep -vq "exit 0"; then echo "chunk job failed"; exit 1; fi
M="nice -n 10 python analyze/v15/merge_chunks_v15.py --delete_chunks"
$M --name v15_s2_c2_p80_half_6000 --first1000 \
   --parts $(for s in 0 750 1500 2250 3000 3750 4500 5250; do echo -n "train:$s:750:chunk "; done) test:0:300:chunk \
   --check $K/v15_s2_c2_p80_half_6000_check_a.h5,$K/v15_s2_c2_p80_half_6000_check_b.h5,train:0 > logs/v15/merge_s2data_A.log 2>&1; echo "A merge exit $?"
$M --name v15_s1_random_half_3000 --first1000 \
   --parts $(for s in 0 750 1500 2250; do echo -n "train:$s:750:chunk "; done) test:0:300:chunk \
   --check $K/v15_s1_random_half_3000_check_a.h5,$K/v15_s1_random_half_3000_check_b.h5,train:0 > logs/v15/merge_s2data_B.log 2>&1; echo "B merge exit $?"
$M --name v15_s1_p80_half_6000 \
   --parts train:0:3000:file=v15_s1_p80_half_3000 $(for s in 3000 3750 4500 5250; do echo -n "train:$s:750:chunk "; done) test:0:300:file=v15_s1_p80_half_3000 \
   --check $K/v15_s1_p80_half_6000_check_a.h5,$K/v15_s1_p80_half_6000_check_b.h5,train:3000 > logs/v15/merge_s2data_C.log 2>&1; echo "C merge exit $?"
nice -n 10 python - > logs/v15/merge_s2data_D.log 2>&1 <<'PY'
import sys; sys.path.insert(0, 'analyze/v15')
from merge_chunks_v15 import write, D
import os, h5py
src = os.path.join(D, 'v15_s1_p80_half_3000', 'data.h5')
attrs = dict(h5py.File(src, 'r').attrs)
write(os.path.join(D, 'v15_s1_p80_half_1000t', 'data.h5'), attrs,
      {'train': [(src, 'train', 0, False, 0, 1000)], 'test': [(src, 'test', 0, False, 0, 300)]}, 'v15_s1_p80_half_1000t')
PY
echo "D exit $?"
echo ALL_DONE
