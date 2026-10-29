#!/bin/bash
# Paper round 11 E1 (docs/paper_extra_log.md Sec.35.4): when the training chain of a pair ends
# (E1_signrand50/s1_signrand50_l1_s<s>/chain_seed<s>.done), run run_e1_eval.sh on the same GPU.
# Usage INSIDE the container, detached:   bash after_e1_train.sh "0:2 1:3 2:4 3:5 4:6 5:7"
# Log: E1_signrand50/eval/logs/launch_e1.log ; all done: E1_signrand50/eval/logs/e1_all.done
cd "$(dirname "$0")/../.."
R=data/result/paper_extra/E1_signrand50
L=$R/eval/logs; mkdir -p $L
log() { echo "$(date '+%m-%d %H:%M:%S %Z') $*" >> $L/launch_e1.log; }
for sg in $1; do
    (s=${sg%%:*}; g=${sg##*:}
     f=$R/s1_signrand50_l1_s$s/chain_seed$s.done
     until [ -f $f ]; do sleep 120; done
     log "signrand50 s$s training $(cat $f); GPU$g $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader); mse $(ls $R/s1_signrand50_mse/$s/model | wc -l) ckpts, l1 $(ls $R/s1_signrand50_l1_s$s/$s/model | wc -l) ckpts"
     GPU=$g bash analyze/paper_extra/run_e1_eval.sh $s
     log "signrand50 s$s eval $(cat $L/signrand50_s$s.done)") &
done
wait
log "all E1 evaluations finished"
touch $L/e1_all.done
