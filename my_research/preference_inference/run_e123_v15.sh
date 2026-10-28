#!/bin/bash
# v15 Sec.36: extract E1-E3 inputs from the 15 saved models on CPU (4 at a time, nice 10). Run INSIDE the container.
cd "$(dirname "$0")"
L=data/result/v15/e123/logs; mkdir -p $L
JOBS=""
for c in exp noexp expY exp_NS noexp_NS; do for s in 0 1 2; do JOBS="$JOBS $c:$s"; done; done
for j in $JOBS; do
  c=${j%%:*}; s=${j#*:}
  case $c in *_NS) e=v15_s2p2h_${c%_NS}_v15i_ns;; *) e=v15_s2p2h_${c}_v15i;; esac
  [ $s != 0 ] && e=${e}_s$s
  while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 10; done
  ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/extract_e123_v15.py --exp $e --seed $s > $L/$e.log 2>&1
    echo "$e seed $s exit $? $(date)" >> $L/done ) &
done
wait
echo ALL_DONE >> $L/done
