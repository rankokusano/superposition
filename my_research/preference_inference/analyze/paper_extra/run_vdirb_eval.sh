#!/bin/bash
# Paper round 13 rev2 (V-dir stage B, docs/paper_extra_log.md Sec.37; pre-registered): matching candidates on the
# stage-B set (vdirb, 1200 episodes) for the given conditions, both stages, late 5 (ep160-200). Pairs 1-6 (switch:
# 1-3). With "rep" as a condition: v12mse:mixed:0 ep200 again (_rep, bit-identity check).
# Run INSIDE the container, detached:  GPU=<g> setsid nohup bash analyze/paper_extra/run_vdirb_eval.sh none p50 > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:?}
[ "$GPU" = "0" ] && { echo "GPU0 is not to be used"; exit 1; }
L=data/result/paper_extra/Vdir_B/eval/logs; mkdir -p $L
R="python -u analyze/paper_extra/vdirb_eval.py --sets vdirb --out ../paper_extra/Vdir_B/eval/inverse"
declare -A SPEC=([none]="v12mse:random v12:random" [p50]="v12mse:mixed v12:mixed" [p80]="v13mse:mixed v13:mixed"
                 [sr80]="csrmse:signrand csr:signrand" [sr50]="csr50mse:signrand50 csr50:signrand50"
                 [sw]="v14cmse:mixedpref v14c:mixedpref")
for c in "$@"; do
    if [ "$c" = "rep" ]; then
        $R --models v12mse:mixed:0 --epochs 200 --tag_suffix _rep > $L/rep.log 2>&1
        continue
    fi
    seeds="0 1 2 3 4 5"; [ "$c" = "sw" ] && seeds="0 1 2"
    for s in $seeds; do
        for sp in ${SPEC[$c]}; do
            $R --models $sp:$s > $L/${c}_${sp%%:*}_s$s.log 2>&1
        done
    done
done
echo "tracebacks $(grep -l Traceback $L/*.log | wc -l) $(date '+%m-%d %H:%M')" > $L/done_$(echo "$@" | tr ' ' '_')
