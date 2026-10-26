#!/bin/bash
# Paper extra C-zero seeds 1, 2 (log Sec.21, 22). Replaces after_gpu12_pzero.sh and the pzero seeds-1/2 part of
# after_c_train.sh. Until 10/27 7:30 JST: seed 1 -> GPU1, seed 2 -> GPU2, each when that GPU had no compute process in
# two checks 3 minutes apart. From 7:30: the seeds not yet launched go to GPU4 / GPU5 / GPU7, using a GPU only after the
# sign-randomised evaluation on it has finished (C_signrand/eval/logs/signrand_s<k>.done, k = 0 / 1 / 2) and it had no
# compute process in two checks 3 minutes apart (GPU1/2 are not used after 7:30). run_c_train.sh refuses an existing
# result directory, so a seed is never launched twice. After each launch, the evaluation watcher (after_c_train.sh)
# for that seed and GPU is started. Gives up at 10/27 14:00. Log: C_pzero/launch.log. Run INSIDE the container, detached.
cd "$(dirname "$0")/../.."
L=data/result/paper_extra/C_pzero; mkdir -p $L
log() { echo "$(date '+%m-%d %H:%M:%S %Z') $*" >> $L/launch.log; }
free() { [ "$(nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader | grep -c .)" = 0 ]; }
free2() { free $1 && sleep 180 && free $1; }
launched() { [ -e $L/launched_seed$1 ] || [ -e data/result/paper_extra/C_pzero/s1_pzero_mse/$1 ]; }
go() {   # seed gpu
    log "GPU$2 free twice ($(nvidia-smi -i $2 --query-gpu=memory.used,utilization.gpu --format=csv,noheader)); launching pzero seed $1"
    touch $L/launched_seed$1
    log "$(bash analyze/paper_extra/run_c_train.sh $1 $2 pzero 2>&1)"
    setsid nohup bash analyze/paper_extra/after_c_train.sh pzero "$1:$2" < /dev/null > /dev/null 2>&1 &
    log "eval watcher started for pzero seed $1 on GPU$2"
}
switch=$(date -d '2026-10-27 07:30 JST' +%s); stop=$(date -d '2026-10-27 14:00 JST' +%s)
declare -A EVAL=([4]=0 [5]=1 [7]=2)   # GPU -> sign-randomised seed evaluated on it
log "pzero_launcher started (replaces after_gpu12_pzero.sh)"
while { ! launched 1 || ! launched 2; } && [ $(date +%s) -lt $stop ]; do
    if [ $(date +%s) -lt $switch ]; then
        for sg in 1:1 2:2; do s=${sg%%:*}; g=${sg##*:}; launched $s || { free2 $g && go $s $g; }; done
    else
        for g in 4 5 7; do
            s=""; launched 1 || s=1; [ -z "$s" ] && { launched 2 || s=2; }
            [ -z "$s" ] && break
            [ -f data/result/paper_extra/C_signrand/eval/logs/signrand_s${EVAL[$g]}.done ] || continue
            grep -q "on GPU $g" $L/launch.log 2>/dev/null && continue    # already holds a pzero run
            free2 $g && go $s $g
        done
    fi
    sleep 60
done
{ launched 1 && launched 2; } || log "gave up at 14:00; seed 1 launched: $(launched 1 && echo yes || echo no), seed 2: $(launched 2 && echo yes || echo no)"
log "pzero_launcher finished"
