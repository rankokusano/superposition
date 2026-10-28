#!/bin/bash
# v15 Sec.39 (as Sec.30.8): as soon as an H4 / R1b model's GPU evaluation is done (done_H4 / done_R1b), evaluate it again on
# the CPU (--suffix _cpu; H4 at the last checkpoint ep1500). At most 4 CPU evaluations at a time, nice 10. Stops when both
# launchers have finished (ALL_DONE_H4, and ALL_DONE_R1B or 'R1b not run'). Run INSIDE the container, detached.
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs
JOBS="v15_s2p2h_exp_v15i_o100:0:H4 v15_s2p2h_exp_v15i_o100_s1:1:H4 v15_s2p2h_exp_v15i_o100_s2:2:H4
      v15_s2p2h_noexp_v15i_o100:0:H4 v15_s2p2h_noexp_v15i_o100_s1:1:H4 v15_s2p2h_noexp_v15i_o100_s2:2:H4
      v15_s2p2h_noexp_v15i_r1b:0:R1b v15_s2p2h_noexp_v15i_r1b_s1:1:R1b v15_s2p2h_noexp_v15i_r1b_s2:2:R1b"
ncpu() { local n=0; for p in /proc/[0-9]*; do tr "\0" " " < $p/cmdline 2>/dev/null | grep -q "eval_s2_hide.*--suffix _cpu" && n=$((n+1)); done; echo $n; }
done_jobs=""
while :; do
  left=0
  for j in $JOBS; do
    case " $done_jobs " in *" $j "*) continue;; esac
    exp=${j%%:*}; r=${j#*:}; s=${r%%:*}; k=${r#*:}; D=$L/done_$k
    if [ $k = H4 ]; then ev="analyze/v15/eval_s2_hide_v15.py --epoch 1500"; else ev=analyze/v15/eval_s2_hide_ns_v15.py; fi
    if grep -q "$exp seed $s eval exit 0" $D 2>/dev/null; then
      while [ $(ncpu) -ge 4 ]; do sleep 30; done
      ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python -u $ev --exp $exp --seed $s --suffix _cpu \
          > $L/hidecpu_${exp}_seed$s.log 2>&1; echo "$exp seed $s cpu eval exit $? $(date)" >> ${D}_cpu ) &
      done_jobs="$done_jobs $j"
    elif grep -q "$exp seed $s .*exit [1-9]\|$exp seed $s not started\|R1b not run" $D 2>/dev/null; then
      echo "$exp seed $s cpu eval skipped (GPU run failed or not started) $(date)" >> ${D}_cpu; done_jobs="$done_jobs $j"
    else left=$((left+1)); fi
  done
  [ $left = 0 ] && break
  sleep 60
done
wait
echo ALL_DONE_H4R1B_CPU >> $L/done_H4_cpu
