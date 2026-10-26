#!/bin/bash
# Paper extra C-zero (log Sec.14.5, 18): from 10/27 0:45, launch pzero seed 1 on GPU1 and seed 2 on GPU2, each as soon as
# that GPU has had no compute process in two checks 3 minutes apart (v15 may chain jobs). Gives up at 10/27 14:00.
# Kusano-san allowed launching without waiting for confirmation at night (log Sec.18). Log: C_pzero/launch.log.
# Run INSIDE the container, detached.
cd "$(dirname "$0")/../.."
L=data/result/paper_extra/C_pzero; mkdir -p $L
log() { echo "$(date '+%m-%d %H:%M:%S %Z') $*" >> $L/launch.log; }
free() { [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
start=$(date -d '2026-10-27 00:45 JST' +%s); stop=$(date -d '2026-10-27 14:00 JST' +%s)
while [ $(date +%s) -lt $start ]; do sleep 60; done
log "start watching GPU1, GPU2"
declare -A todo=([1]=1 [2]=2)   # gpu -> seed
while [ ${#todo[@]} -gt 0 ] && [ $(date +%s) -lt $stop ]; do
    for g in "${!todo[@]}"; do
        if free $g; then
            sleep 180
            if free $g; then
                log "GPU$g free twice ($(nvidia-smi -i $g --query-gpu=memory.used,utilization.gpu --format=csv,noheader)); launching pzero seed ${todo[$g]}"
                log "$(bash analyze/paper_extra/run_c_train.sh ${todo[$g]} $g pzero 2>&1)"
                unset "todo[$g]"
            fi
        fi
    done
    sleep 60
done
[ ${#todo[@]} -gt 0 ] && log "gave up at 14:00; not launched: GPU(s) ${!todo[*]}"
log "watcher finished"
