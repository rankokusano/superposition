#!/bin/bash
# v15 Sec.43.4 (descriptive): CPU evaluations of the H4 runs before overfitting, at the epochs fixed in Sec.43.4 (ep200 for all
# 6 runs; the test-loss minimum per run, which is ep200 for three of them). Output hide_<name>_ep<E>_cpu.json.
# At most 4 CPU evaluations at a time (counting the other watchers' ones), nice 10. Run INSIDE the container, detached.
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs
JOBS="v15_s2p2h_exp_v15i_o100:0:200 v15_s2p2h_exp_v15i_o100_s1:1:200 v15_s2p2h_exp_v15i_o100_s2:2:200
      v15_s2p2h_noexp_v15i_o100:0:200 v15_s2p2h_noexp_v15i_o100_s1:1:200 v15_s2p2h_noexp_v15i_o100_s2:2:200
      v15_s2p2h_exp_v15i_o100_s1:1:50 v15_s2p2h_noexp_v15i_o100:0:100 v15_s2p2h_noexp_v15i_o100_s1:1:300"
ncpu() { local n=0; for p in /proc/[0-9]*; do tr "\0" " " < $p/cmdline 2>/dev/null | grep -q "eval_s2_hide.*--suffix _.*cpu" && n=$((n+1)); done; echo $n; }
for j in $JOBS; do
  IFS=: read exp s ep <<< "$j"
  while [ $(ncpu) -ge 4 ]; do sleep 30; done
  ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s2_hide_v15.py --exp $exp --seed $s \
      --epoch $ep --suffix _ep${ep}_cpu > $L/hidecpu_${exp}_seed${s}_ep$ep.log 2>&1
    echo "$exp seed $s ep$ep cpu eval exit $? $(date)" >> $L/done_H4early_cpu ) &
  sleep 5
done
wait
echo ALL_DONE_H4EARLY_CPU >> $L/done_H4early_cpu
