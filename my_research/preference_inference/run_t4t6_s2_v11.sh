#!/bin/bash
# v11 stage-2 T4 / T6 / A-2 tracking (docs/v11_experiment_log.md Sec.7.53): 8 runs x G late-5 on GPU3-7,
# plus the data-only D6 values. Run INSIDE the container from the v11 worktree's
# my_research/preference_inference. GPU0 is never used; GPU1/2 are the s3 stage-1 runs.
set -u
cd "$(dirname "$0")"
L=data/result/v11/stage2/t4t6/logs; mkdir -p $L
go() { local gpu=$1; shift; local name=$1; shift
    setsid nohup bash -c "CUDA_VISIBLE_DEVICES=${gpu} python -u analyze/v11/t4t6_s2_v11.py --runs $* > $L/${name}.log 2>&1; echo \"exit \$?\" > $L/${name}.done" < /dev/null > /dev/null 2>&1 &
    echo "GPU${gpu}: $*"; }
go 3 g3 green:random:0 green:mixed:0
go 4 g4 pref:random:0 pref:mixed:0
go 5 g5 green:random:2 green:mixed:2
go 6 g6 pref:random:2
go 7 g7 pref:mixed:2
