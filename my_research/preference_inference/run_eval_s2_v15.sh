#!/bin/bash
# v15 stage-2 evaluation (Sec.16, 16.7, 19): for each tag, wait until its phase-2 training wrote done_seed0 with exit 0,
# then run eval_s2_v15.py on the GPU that trained it (free by then). Args: pairs tag:gpu ... Run INSIDE the container.
set -u
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs; mkdir -p $L
one() { local tag=${1%%:*} gpu=${1#*:}
  until [ -f data/result/v15_s2p2_${tag}/done_seed0 ]; do sleep 120; done
  grep -q "exit 0" data/result/v15_s2p2_${tag}/done_seed0 || { echo "$tag phase 2 failed, no eval" >> $L/eval_done; return; }
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s2_v15.py --tag $tag --device cuda:0 \
    > $L/eval_${tag}.log 2>&1
  rc=$?; echo "$tag eval exit $rc $(date)" >> $L/eval_done
}
for a in "$@"; do one $a & done
wait
echo EVAL_ALL_DONE >> $L/eval_done
