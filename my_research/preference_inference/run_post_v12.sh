#!/bin/bash
# v12 post-hoc analysis B (docs/v12_experiment_log.md Sec.41) on GPU3-6 (never GPU0). Detached, inside the container.
set -u
cd "$(dirname "$0")"
L=data/result/v12/post/logs; mkdir -p $L
run() { local gpu=$1; shift; for job in "$@"; do set -- $job
  CUDA_VISIBLE_DEVICES=$gpu python -u analyze/v12/s2_post_v12.py --prefix $1 --seed $2 > $L/$1_s$2.log 2>&1; echo "$1 s$2 exit $? $(date)" >> $L/done
done; }
run 3 "v12_s2 0" "v12_s2 3" &
run 4 "v12_s2 1" "v12_s2b 0" &
run 5 "v12_s2 2" "v12_s2b 2" &
wait
echo "all done $(date)" >> $L/done
