#!/bin/bash
# v15 Sec.30.8: as soon as a (A)/(B) model's GPU evaluation is done (done_AB), evaluate it again on the CPU (same computation as
# seed 0: eval_s2_hide_v15.py --suffix _cpu). At most 4 CPU evaluations at a time, nice 10. Run INSIDE the container, detached.
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs
JOBS="v15_s2p2h_expY_v4:0 v15_s2p2h_expY_v15i:0 v15_s2p2h_exp_v4_s1:1 v15_s2p2h_noexp_v4_s1:1 v15_s2p2h_exp_v4_s2:2 v15_s2p2h_noexp_v4_s2:2
      v15_s2p2h_exp_v15i_s1:1 v15_s2p2h_noexp_v15i_s1:1 v15_s2p2h_exp_v15i_s2:2 v15_s2p2h_noexp_v15i_s2:2"
ncpu() { local n=0; for p in /proc/[0-9]*; do tr "\0" " " < $p/cmdline 2>/dev/null | grep -q "eval_s2_hide_v15.py.*--suffix _cpu" && n=$((n+1)); done; echo $n; }
done_jobs=""
while :; do
  left=0
  for j in $JOBS; do
    case " $done_jobs " in *" $j "*) continue;; esac
    exp=${j%%:*}; s=${j#*:}
    if grep -q "$exp seed $s eval exit 0" $L/done_AB 2>/dev/null; then
      while [ $(ncpu) -ge 4 ]; do sleep 30; done
      ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s2_hide_v15.py --exp $exp --seed $s --suffix _cpu \
          > $L/hidecpu_${exp}_seed$s.log 2>&1; echo "$exp seed $s cpu eval exit $? $(date)" >> $L/done_AB_cpu ) &
      done_jobs="$done_jobs $j"
    elif grep -q "$exp seed $s .*exit [1-9]\|$exp seed $s not started" $L/done_AB 2>/dev/null; then
      echo "$exp seed $s skipped (GPU run failed or not started) $(date)" >> $L/done_AB_cpu; done_jobs="$done_jobs $j"
    else
      left=1
    fi
  done
  [ $left = 0 ] && break
  sleep 120
done
wait
echo ALL_DONE_CPU >> $L/done_AB_cpu
