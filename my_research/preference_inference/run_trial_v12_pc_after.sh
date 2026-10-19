#!/bin/bash
# v12 trial (P-c), after training (docs/v12_experiment_log.md Sec.10): Fig.5 Decoders (GPU3/4),
# evaluation ep60/100/150 (GPU5), D6-like gate ep150 (GPU6), repeat evaluation of ep150 for the
# bit-identity check (GPU7), encoder check, then Fig.5 read-outs. Run INSIDE the container from the
# v11 worktree's my_research/preference_inference. GPU0 never used; GPU1/2 stay with v11 s3.
set -u
cd "$(dirname "$0")"
L=data/result/v12/trial/logs; mkdir -p $L
until [ -f data/result/v12_trial_pc_random_mse/done_seed0 ] && [ -f data/result/v12_trial_pc_mixed_mse/done_seed0 ]; do sleep 60; done
i=3
for c in random mixed; do
    CUDA_VISIBLE_DEVICES=$i python -u train_v12.py --exp_config v12_trial_fig5_pc_${c} --seed 0 --device cuda:0 \
        --cudnn_deterministic --data_load_memory > $L/fig5_train_pc_${c}.log 2>&1 &
    i=$((i+1))
done
(CUDA_VISIBLE_DEVICES=5 python -u analyze/v12/eval_trial_v12.py --exp_fmt v12_trial_pc_{}_mse --prefix pc_ --epochs 60 100 150 > $L/pc_eval.log 2>&1; echo "exit $?" > $L/pc_eval.done) &
(CUDA_VISIBLE_DEVICES=6 python -u analyze/v12/eval_d6_v12.py --exp_fmt v12_trial_pc_{}_mse --sub trial/d6_pc > $L/d6_pc.log 2>&1; echo "exit $?" > $L/d6_pc.done) &
(CUDA_VISIBLE_DEVICES=7 python -u analyze/v12/eval_trial_v12.py --exp_fmt v12_trial_pc_{}_mse --prefix pc_ --epochs 150 --tag _rep > $L/pc_eval_rep.log 2>&1; echo "exit $?" > $L/pc_eval_rep.done) &
python - > $L/pc_encoder_check.log 2>&1 <<'PY'
import torch
ref = torch.load('/work/data/result/exp1_l1_1000/0/model/00200.pth', map_location='cpu')['model']
for c in ('random', 'mixed'):
    sd = torch.load(f'data/result/v12_trial_pc_{c}_mse/0/model/00150.pth', map_location='cpu')['model']
    ks = [k for k in ref if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module', 'share_lns'))]
    print(c, len(ks), 'all equal to v4:', all(torch.equal(sd[k], ref[k]) for k in ks))
PY
wait
for c in random mixed; do
    CUDA_VISIBLE_DEVICES=3 python -u analyze/v11/fig5_v11.py --ae v12_trial_fig5_pc_${c} --seed 0 --epoch 100 --sub ../v12/trial/fig5_pc_${c} > $L/fig5_eval_pc_${c}.log 2>&1
    echo "fig5 pc ${c} exit $?" >> $L/fig5_pc.done
done
