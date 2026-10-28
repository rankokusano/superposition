#!/bin/bash
# v15 Sec.41 (as Sec.30.8): as soon as an R2b model's GPU evaluation is done (done_R2b), evaluate it again on the CPU
# (--suffix _cpu). At most 4 CPU evaluations at a time, nice 10. Run INSIDE the container, detached.
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs
JOBS="v15_s2p2h_noexp_v15i_r2b:0 v15_s2p2h_noexp_v15i_r2b_s1:1 v15_s2p2h_noexp_v15i_r2b_s2:2"
ncpu() { local n=0; for p in /proc/[0-9]*; do tr "\0" " " < $p/cmdline 2>/dev/null | grep -q "eval_s2_hide.*--suffix _cpu" && n=$((n+1)); done; echo $n; }
done_jobs=""
while :; do
  left=0
  for j in $JOBS; do
    case " $done_jobs " in *" $j "*) continue;; esac
    exp=${j%%:*}; s=${j#*:}
    if grep -q "$exp seed $s eval exit 0" $L/done_R2b 2>/dev/null; then
      while [ $(ncpu) -ge 4 ]; do sleep 30; done
      ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s2_hide_ns_v15.py --exp $exp --seed $s \
          --suffix _cpu > $L/hidecpu_${exp}_seed$s.log 2>&1; echo "$exp seed $s cpu eval exit $? $(date)" >> $L/done_R2b_cpu ) &
      done_jobs="$done_jobs $j"
    elif grep -q "$exp seed $s .*exit [1-9]\|$exp seed $s not started" $L/done_R2b 2>/dev/null; then
      echo "$exp seed $s cpu eval skipped (GPU run failed or not started) $(date)" >> $L/done_R2b_cpu; done_jobs="$done_jobs $j"
    else left=$((left+1)); fi
  done
  [ $left = 0 ] && break
  sleep 60
done
wait
echo ALL_DONE_R2B_CPU >> $L/done_R2b_cpu
