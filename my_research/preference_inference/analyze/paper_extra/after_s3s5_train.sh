#!/bin/bash
# Paper round 6 (log Sec.34.4, 34.6): when the signrand training chain of a pair ends
# (C_signrand/s1_signrand_l1_s<s>/chain_seed<s>.done), run on the same GPU the C evaluation set (run_c_eval.sh, which
# also runs g3flip for the pair's 80% and no-experience runs) and then the B4 set (run_b4.sh: 80%, signrand, none).
# Usage INSIDE the container, detached:   bash after_s3s5_train.sh "3:1 4:2 5:4"
# Log: C_signrand/eval/logs/launch_s3s5.log
cd "$(dirname "$0")/../.."
L=data/result/paper_extra/C_signrand/eval/logs; mkdir -p $L
log() { echo "$(date '+%m-%d %H:%M:%S %Z') $*" >> $L/launch_s3s5.log; }
test -f data/result/paper_extra/B4/logs/data.done || { log "B4 data.done missing"; exit 1; }
for sg in $1; do
    (s=${sg%%:*}; g=${sg##*:}
     f=data/result/paper_extra/C_signrand/s1_signrand_l1_s$s/chain_seed$s.done
     until [ -f $f ]; do sleep 120; done
     log "signrand s$s training $(cat $f); GPU$g $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader); mse $(ls data/result/paper_extra/C_signrand/s1_signrand_mse/$s/model | wc -l) ckpts, l1 $(ls data/result/paper_extra/C_signrand/s1_signrand_l1_s$s/$s/model | wc -l) ckpts"
     GPU=$g bash analyze/paper_extra/run_c_eval.sh signrand $s
     log "signrand s$s C eval $(cat $L/signrand_s$s.done)"
     GPU=$g bash analyze/paper_extra/run_b4.sh $s
     log "s$s B4 eval $(cat data/result/paper_extra/B4/logs/eval_s$s.done)") &
done
wait
log "all s3s5 evaluations finished"
touch $L/s3s5_all.done
