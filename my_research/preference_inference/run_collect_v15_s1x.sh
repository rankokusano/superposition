#!/bin/bash
# v15 stage-1 data extension (docs/v15_experiment_log.md Sec.8): v15_s1_p80 -> train 3000 + test 300, same construction
# (key v15_s1). Chunks into data/data/v15_s1_p80_3000/chunks/, merged by analyze/v15/merge_s1x_v15.py.
# Plus an overlap check: train episodes 980..999 regenerated and compared with v15_s1_p80. CPU only, nice 10, 4 at a time.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= MESA_GL_VERSION_OVERRIDE=3.3
C=data/data/v15_s1_p80_3000/chunks; mkdir -p $C logs/v15
run_one() {
  set -- $1
  nice -n 10 xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' python simulation/collect_data_v15.py \
    --dataset v15_s1_p80 --split $1 --start $2 --n $3 --save_path data/data/v15_s1_p80_3000/chunks/$1_$2.h5 \
    > logs/v15/collect_s1x_$1_$2.log 2>&1
  echo "$(date +%T) done $1 $2 $3 (exit $?)"
}
export -f run_one
printf '%s\n' "train 1000 500" "train 1500 500" "train 2000 500" "train 2500 500" "test 0 300" "train 980 20" \
  | xargs -P 4 -I{} bash -c 'run_one "{}"'
echo ALL_DONE
