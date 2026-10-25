#!/bin/bash
# v15 stage-1 data with gray in half of the episodes (docs/v15_experiment_log.md Sec.12): train 3000 + test 300 in chunks,
# plus train 0..19 generated twice (reproducibility), then merge (analyze/v15/merge_s1half_v15.py). CPU only, nice 10,
# 4 at a time. Run INSIDE the container, detached.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= MESA_GL_VERSION_OVERRIDE=3.3
C=data/data/v15_s1_p80_half_3000/chunks; K=data/result/v15/datacheck; mkdir -p $C $K logs/v15
run_one() {
  set -- $1
  if [ "$1" = check ]; then out=data/result/v15/datacheck/v15_s1_p80_half_train0_check_$2.h5; args="--split train --start 0 --n 20"
  else out=data/data/v15_s1_p80_half_3000/chunks/$1_$2.h5; args="--split $1 --start $2 --n $3"; fi
  nice -n 10 xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' python simulation/collect_data_v15.py \
    --dataset v15_s1_p80_half $args --save_path $out > logs/v15/collect_s1half_$1_$2.log 2>&1
  rc=$?; echo "$(date +%T) done $* (exit $rc)"
}
export -f run_one
printf '%s\n' "train 0 750" "train 750 750" "train 1500 750" "train 2250 750" "test 0 300" "check a" "check b" \
  | xargs -P 4 -I{} bash -c 'run_one "{}"'
if grep -v "exit 0" logs/v15_collect_s1half_driver.log | grep -q done; then echo "chunk job failed"; exit 1; fi
nice -n 10 python analyze/v15/merge_s1half_v15.py --delete_chunks > logs/v15/merge_s1half.log 2>&1
rc=$?; echo "$(date +%T) merge exit $rc"
echo ALL_DONE
