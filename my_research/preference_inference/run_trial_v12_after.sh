#!/bin/bash
# v12 trial, after training (docs/v12_experiment_log.md Sec.4): on GPU7, wait for both trial runs,
# then (a) Fig.5 Decoders for both conditions and (b) evaluation at ep100 and ep150, in parallel;
# then the Fig.5 read-out. Run INSIDE the container from the v11 worktree's my_research/preference_inference.
set -u
cd "$(dirname "$0")"
L=data/result/v12/trial/logs; mkdir -p $L
until [ -f data/result/v12_trial_random_mse/done_seed0 ] && [ -f data/result/v12_trial_mixed_mse/done_seed0 ]; do sleep 60; done
for c in random mixed; do
    CUDA_VISIBLE_DEVICES=7 python -u train_v12.py --exp_config v12_trial_fig5_${c} --seed 0 --device cuda:0 \
        --cudnn_deterministic --data_load_memory > $L/fig5_train_${c}.log 2>&1 &
done
CUDA_VISIBLE_DEVICES=7 python -u analyze/v12/eval_trial_v12.py --epochs 100 150 > $L/eval_ep100_150.log 2>&1
echo "eval exit $?" > $L/eval_ep100_150.done
wait
for c in random mixed; do
    CUDA_VISIBLE_DEVICES=7 python -u analyze/v11/fig5_v11.py --ae v12_trial_fig5_${c} --seed 0 --epoch 100 --sub ../v12/trial/fig5 > $L/fig5_eval_${c}.log 2>&1
    echo "fig5 ${c} exit $?" >> $L/fig5.done
done
