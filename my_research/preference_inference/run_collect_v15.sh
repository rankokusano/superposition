#!/bin/bash
# v15 stage-0 data (docs/v15_experiment_log.md Sec.4.1). CPU only, nice 10, 4 jobs at a time.
# Run inside Docker from the v15 worktree's my_research/preference_inference.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= MESA_GL_VERSION_OVERRIDE=3.3
LOG=logs/v15; mkdir -p $LOG data/result/v15/datacheck
MAIN="v15_s1_p80 v15_s2_c2_p80 v15_s2_genx_p80 v15_s2_geny_p80 v15_s2_c3x_p80 v15_s2_c3xp_p80 v15_s2_c1_p80
      v15_s2_c2_p90 v15_s2_c1_p90 v15_s1_p50 v15_s2_c2_p50 v15_s2_c1_p50 v15_s2_c3x_p50 v15_s2_c3xp_p50"
JOBS=""
for d in $MAIN; do JOBS="$JOBS $d:"; done
JOBS="$JOBS v15_s1_p80:full_repeat v15_s2_c2_p80:full_repeat"
for d in $MAIN; do case $d in v15_s1_p80|v15_s2_c2_p80) ;; *) JOBS="$JOBS ${d}_check_a: ${d}_check_b:";; esac; done
run_one() {
  d=${1%%:*}; tag=${1#*:}
  if [ "$tag" = full_repeat ]; then
    nice -n 10 xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' python simulation/collect_data_v15.py \
      --dataset $d --save_path data/result/v15/datacheck/${d}_full_repeat.h5 > logs/v15/collect_${d}_full_repeat.log 2>&1
  else
    nice -n 10 xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' python simulation/collect_data_v15.py \
      --dataset $d > logs/v15/collect_${d}.log 2>&1
  fi
  echo "$(date +%T) done $1 (exit $?)"
}
export -f run_one
echo $JOBS | tr ' ' '\n' | grep -v '^$' | xargs -P 4 -I{} bash -c 'run_one {}'
echo ALL_DONE
