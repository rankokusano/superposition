#!/bin/bash
# Paper extra B (log Sec.13, 16): wait until the copy-model evaluation (run_b_eval.sh, seeds 0-2) has finished,
# check that GPU6 has no compute process left, then start run_b_eval_pcols.sh for seeds 0-2 in parallel on GPU6.
# Writes data/result/paper_extra/B_copy_pcols/logs/launch.log. Run INSIDE the container, detached.
cd "$(dirname "$0")/../.."
L=data/result/paper_extra/B_copy_pcols/logs; mkdir -p $L
B=data/result/paper_extra/B_copy/logs
until [ -f $B/seed0.done ] && [ -f $B/seed1.done ] && [ -f $B/seed2.done ]; do sleep 60; done
sleep 30
n=$(nvidia-smi -i 6 --query-compute-apps=pid --format=csv,noheader | grep -c .)
echo "$(date) B done ($(cat $B/seed*.done | tr '\n' ' ')); GPU6 compute processes: $n; $(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader)" >> $L/launch.log
if [ "$n" != "0" ]; then echo "$(date) GPU6 busy: NOT launched" >> $L/launch.log; exit 1; fi
for s in 0 1 2; do GPU=6 bash analyze/paper_extra/run_b_eval_pcols.sh $s & done
echo "$(date) launched pcols seeds 0-2 on GPU6" >> $L/launch.log
wait
echo "$(date) pcols finished" >> $L/launch.log
