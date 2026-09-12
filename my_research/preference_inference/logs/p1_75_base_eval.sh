#!/bin/bash
set -u
cd /work/my_research/preference_inference
export CUDA_VISIBLE_DEVICES=7
export DEVICE=cuda:0
running(){ for p in /proc/[0-9]*; do [ -r "$p/cmdline" ] || continue; tr "\0" " " < "$p/cmdline" 2>/dev/null | grep -q "p1_75_dir_fix.sh" && return 0; done; return 1; }
echo "[base-eval] waiting for dir-fix ..."
while running; do sleep 30; done
echo "[base-eval] $(date -u) eval r3_stay_400/0 ep400 on r2_a1red_a2red (base h2->self reference)"
bash analyze/run_eval_v4v5.sh r3_stay_400 400 0 r2_a1red_a2red 0 2>&1 | grep -vE "it/s\]|it\]" | grep -E "h1 ->|h2 ->|R\^2|Saved:|DONE"
echo "[base-eval] DONE"
