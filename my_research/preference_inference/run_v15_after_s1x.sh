#!/bin/bash
# v15 (docs/v15_experiment_log.md Sec.8.2, 9): after run_collect_v15_s1x.sh -> merge v15_s1_p80_3000 (overlap check first;
# chunks deleted only after a successful merge) -> stage-2 proposal data (8 x 300 episodes, nice 10, 4 at a time)
# -> proposal analysis (OMP 2 threads). CPU only. Run INSIDE the container, detached.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= MESA_GL_VERSION_OVERRIDE=3.3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
until grep -q ALL_DONE logs/v15_collect_s1x_driver.log 2>/dev/null; do sleep 120; done
grep -v "exit 0" logs/v15_collect_s1x_driver.log | grep -v ALL_DONE && { echo "chunk job failed"; exit 1; }
nice -n 10 python analyze/v15/merge_s1x_v15.py --delete_chunks > logs/v15/merge_s1x.log 2>&1 || { echo "merge failed"; exit 1; }
echo "$(date +%T) merged"
run_one() {
  nice -n 10 xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' python simulation/collect_data_v15.py \
    --dataset $1 > logs/v15/collect_$1.log 2>&1
  rc=$?; echo "$(date +%T) done $1 (exit $rc)"
}
export -f run_one
printf '%s\n' v15p_i_grayseq_p80 v15p_i_grayhalf_p80 v15p_ii_p30 v15p_ii_p40 v15p_ii_p50 v15p_ii_p60 v15p_ii_p80 v15p_ii_p50_T151 \
  | xargs -P 4 -I{} bash -c 'run_one {}'
nice -n 10 python -u analyze/v15/proposals_v15.py > logs/v15/proposals.log 2>&1
rc=$?; echo "$(date +%T) proposals exit $rc"
echo ALL_DONE
