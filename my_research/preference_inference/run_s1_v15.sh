#!/bin/bash
# v15 stage 1 (docs/v15_experiment_log.md Sec.8, 12): two runs on GPU4 only (never GPU0), seed 0:
#   (i) v15_s1_plain_{mse,l1}, (ii) v15_s1_colour_{mse,l1}; mse 200 ep -> l1 200 ep (Decoder only).
# Evaluation chain per run (also GPU4): at mse ep100/150/200 train the Fig.5 Decoder (100 ep), run fig5_v11.py,
# then eval_s1_v15.py (CPU); after l1: check the encoders equal mse ep200 and evaluate l1 ep200 (encoder measures).
# Run INSIDE the container, detached.
set -u
cd "$(dirname "$0")"
GPU=4
L=data/result/v15/stage1/logs; mkdir -p $L
train() { local loss=$1
  for st in mse l1; do exp=v15_s1_${loss}_${st}; mkdir -p data/result/$exp
    CUDA_VISIBLE_DEVICES=$GPU python -u train_v15.py --exp_config $exp --seed 0 --device cuda:0 --cudnn_deterministic \
      --data_load_memory > data/result/$exp/train_seed0.log 2>&1
    echo "exit $?" > data/result/$exp/done_seed0
    echo "$exp train exit $(cat data/result/$exp/done_seed0) $(date)" >> $L/done
  done; }
wait_ck() { until [ -f data/result/$1/0/model/$(printf %05d $2).pth ]; do sleep 120; done; sleep 30; }
evalchain() { local loss=$1 exp=v15_s1_${1}_mse
  for e in 100 150 200; do
    wait_ck $exp $e
    ae=v15_s1_${loss}_fig5_mse$e
    CUDA_VISIBLE_DEVICES=$GPU python -u train_v12.py --exp_config $ae --seed 0 --device cuda:0 --cudnn_deterministic \
      --data_load_memory > $L/${ae}_train.log 2>&1
    echo "$ae train exit $? $(date)" >> $L/done
    CUDA_VISIBLE_DEVICES=$GPU python -u analyze/v11/fig5_v11.py --ae $ae --seed 0 --epoch 100 \
      --sub ../v15/stage1/fig5_${loss}_mse$e > $L/${ae}_fig5.log 2>&1
    echo "$ae fig5 exit $? $(date)" >> $L/done
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s1_v15.py --tag v15_${loss}_mse$e \
      --ckpt data/result/$exp/0/model/$(printf %05d $e).pth --ae data/result/$ae/0/model/00100.pth > $L/${ae}_eval.log 2>&1
    echo "$ae eval exit $? $(date)" >> $L/done
  done
  until [ -f data/result/v15_s1_${loss}_l1/done_seed0 ]; do sleep 300; done
  python - $exp v15_s1_${loss}_l1 >> $L/done 2>&1 <<'PY'
import sys, torch
a = torch.load(f'data/result/{sys.argv[1]}/0/model/00200.pth', map_location='cpu')['model']
b = torch.load(f'data/result/{sys.argv[2]}/0/model/00200.pth', map_location='cpu')['model']
ks = [k for k in a if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module', 'share_lns'))]
print(sys.argv[2], 'encoders identical to mse ep200:', all(torch.equal(a[k], b[k]) for k in ks), len(ks))
PY
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s1_v15.py --tag v15_${loss}_l1_200 \
    --ckpt data/result/v15_s1_${loss}_l1/0/model/00200.pth > $L/v15_${loss}_l1_eval.log 2>&1
  echo "v15_${loss}_l1 eval exit $? $(date)" >> $L/done
}
train plain &
train colour &
evalchain plain &
evalchain colour &
wait
echo ALL_DONE >> $L/done
