#!/bin/bash
# Paper extra C (log Sec.18): when a training chain ends (data/result/paper_extra/<dir>/s1_<c>_l1_s<s>/chain_seed<s>.done),
# start run_c_eval.sh for that run on the same GPU. Usage INSIDE the container, detached:
#     bash after_c_train.sh <signrand|pzero> "<seed>:<gpu> ..."      Log: C_signrand/eval/logs/launch.log
cd "$(dirname "$0")/../.."
c=$1; pairs=$2
d=$([ $c = signrand ] && echo C_signrand || echo C_pzero)
L=data/result/paper_extra/C_signrand/eval/logs; mkdir -p $L
log() { echo "$(date '+%m-%d %H:%M:%S %Z') $*" >> $L/launch.log; }
for sg in $pairs; do
    (s=${sg%%:*}; g=${sg##*:}
     f=data/result/paper_extra/$d/s1_${c}_l1_s$s/chain_seed$s.done
     until [ -f $f ]; do sleep 120; done
     log "$c s$s training $(cat $f); GPU$g $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader), procs $(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | grep -c .); l1 ep200 $(ls data/result/paper_extra/$d/s1_${c}_l1_s$s/$s/model/00200.pth 2>&1)"
     GPU=$g bash analyze/paper_extra/run_c_eval.sh $c $s
     log "$c s$s eval $(cat $L/${c}_s$s.done)") &
done
wait
log "$c all evaluations finished"
