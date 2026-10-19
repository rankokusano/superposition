#!/bin/bash
# v12 trials (P-a)/(P-b), after training (docs/v12_experiment_log.md Sec.8): Fig.5 Decoders (GPU3/4/5),
# evaluation (P-a on GPU3, P-b on GPU6), encoder-unchanged check for P-b, then Fig.5 read-outs.
# Run INSIDE the container from the v11 worktree's my_research/preference_inference. GPU0 never used.
set -u
cd "$(dirname "$0")"
L=data/result/v12/trial/logs; mkdir -p $L
until [ -f data/result/v12_trial_pa_random_mse/done_seed0 ] && [ -f data/result/v12_trial_pb_random_mse/done_seed0 ] && [ -f data/result/v12_trial_pb_mixed_mse/done_seed0 ]; do sleep 60; done
i=3
for x in pa_random pb_random pb_mixed; do
    CUDA_VISIBLE_DEVICES=$i python -u train_v12.py --exp_config v12_trial_fig5_${x} --seed 0 --device cuda:0 \
        --cudnn_deterministic --data_load_memory > $L/fig5_train_${x}.log 2>&1 &
    i=$((i+1))
done
(CUDA_VISIBLE_DEVICES=3 python -u analyze/v12/eval_trial_pa_v12.py --epochs 60 100 150 > $L/pa_eval.log 2>&1; echo "exit $?" > $L/pa_eval.done) &
(CUDA_VISIBLE_DEVICES=6 python -u analyze/v12/eval_trial_v12.py --exp_fmt v12_trial_pb_{}_mse --prefix pb_ --epochs 60 100 150 > $L/pb_eval.log 2>&1; echo "exit $?" > $L/pb_eval.done) &
python - > $L/pb_encoder_check.log 2>&1 <<'PY'
import torch
ref = torch.load('/work/data/result/exp1_l1_1000/0/model/00200.pth', map_location='cpu')['model']
for c in ('random', 'mixed'):
    sd = torch.load(f'data/result/v12_trial_pb_{c}_mse/0/model/00150.pth', map_location='cpu')['model']
    ks = [k for k in ref if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module', 'share_lns'))]
    print(c, len(ks), 'all equal to v4:', all(torch.equal(sd[k], ref[k]) for k in ks))
PY
wait
for x in pa_random pb_random pb_mixed; do
    CUDA_VISIBLE_DEVICES=3 python -u analyze/v11/fig5_v11.py --ae v12_trial_fig5_${x} --seed 0 --epoch 100 --sub ../v12/trial/fig5_${x} > $L/fig5_eval_${x}.log 2>&1
    echo "fig5 ${x} exit $?" >> $L/fig5_pab.done
done
