#!/bin/bash
# v12 stage 2c dry run (docs/v12_experiment_log.md Sec.46, NOT a result): hidden-test evaluation scripts on the
# trained stage-2b pair s0 (GPU3-5, never GPU0). Detached, inside the container.
set -u
cd "$(dirname "$0")"
L=data/result/v12/stage2c_dry/logs; mkdir -p $L
(CUDA_VISIBLE_DEVICES=3 python -u analyze/v12/eval_s2c_v12.py --prefix v12_s2b --out stage2c_dry --runs green:random:0 green:mixed:0 pref:random:0 pref:mixed:0 --epochs 200 > $L/eval1.log 2>&1
 mkdir -p data/result/v12/stage2c_dry/rep; cp data/result/v12/stage2c_dry/eval/green_mixed_s0_ep200* data/result/v12/stage2c_dry/rep/
 CUDA_VISIBLE_DEVICES=3 python -u analyze/v12/eval_s2c_v12.py --prefix v12_s2b --out stage2c_dry --runs green:mixed:0 --epochs 200 > $L/eval2.log 2>&1
 CUDA_VISIBLE_DEVICES=3 python -u analyze/v12/eval_s2c_v12.py --out stage2c_dry --runs d6:data:0 > $L/d6.log 2>&1
 echo "eval exit $? $(date)" >> $L/done) &
(CUDA_VISIBLE_DEVICES=4 python -u analyze/v12/eval_s2c_b3_v12.py --prefix v12_s2b --out stage2c_dry_orig --setting green --seed 0 --epochs 200 --input_key self_vision > $L/b3_orig.log 2>&1
 echo "b3_orig exit $? $(date)" >> $L/done) &
(CUDA_VISIBLE_DEVICES=5 python -u analyze/v12/eval_s2c_b3_v12.py --prefix v12_s2b --out stage2c_dry --setting green --seed 0 --epochs 200 > $L/b3.log 2>&1
 echo "b3 exit $? $(date)" >> $L/done) &
wait
