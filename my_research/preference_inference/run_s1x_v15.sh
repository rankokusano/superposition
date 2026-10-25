#!/bin/bash
# v15 Sec.15 stage-1 extra exploration (NOT pre-registered; same measures and criteria as Sec.8.4-8.5):
#   v15_s1_colourl1 (colour_lambda 1) on GPU $G1, v15_s1_colourl3 (lambda 3) on GPU $G2,
#   v15_s1_plain6k ((i) on v15_s1_p80_half_6000) on GPU $G3 after that dataset exists.
# Per run: mse 200 ep -> l1 200 ep; at mse ep100/150/200: Fig.5 Decoder, fig5_v11.py, eval_s1_v15.py; after l1: encoder check.
# Usage (INSIDE the container, detached): ./run_s1x_v15.sh <G1> <G2> <G3>
set -u
cd "$(dirname "$0")"
L=data/result/v15/stage1/logs; mkdir -p $L
wait_ck() { until [ -f data/result/$1/0/model/$(printf %05d $2).pth ]; do sleep 120; done; sleep 30; }
one() { local run=$1 gpu=$2
  if [ $run = plain6k ]; then until [ -f data/data/v15_s1_p80_half_6000/data.h5 ] && grep -q "C merge exit 0" logs/v15_s2data_driver.log 2>/dev/null; do sleep 300; done; fi
  ( for st in mse l1; do exp=v15_s1_${run}_${st}; mkdir -p data/result/$exp
      CUDA_VISIBLE_DEVICES=$gpu python -u train_v15.py --exp_config $exp --seed 0 --device cuda:0 --cudnn_deterministic \
        --data_load_memory > data/result/$exp/train_seed0.log 2>&1
      echo "exit $?" > data/result/$exp/done_seed0
      echo "$exp train $(cat data/result/$exp/done_seed0) $(date)" >> $L/done_x
    done ) &
  local exp=v15_s1_${run}_mse
  for e in 100 150 200; do
    wait_ck $exp $e
    ae=v15_s1_${run}_fig5_mse$e
    CUDA_VISIBLE_DEVICES=$gpu python -u train_v12.py --exp_config $ae --seed 0 --device cuda:0 --cudnn_deterministic \
      --data_load_memory > $L/${ae}_train.log 2>&1
    echo "$ae train exit $? $(date)" >> $L/done_x
    CUDA_VISIBLE_DEVICES=$gpu python -u analyze/v11/fig5_v11.py --ae $ae --seed 0 --epoch 100 \
      --sub ../v15/stage1/fig5_${run}_mse$e > $L/${ae}_fig5.log 2>&1
    echo "$ae fig5 exit $? $(date)" >> $L/done_x
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 python -u analyze/v15/eval_s1_v15.py --tag v15_${run}_mse$e \
      --ckpt data/result/$exp/0/model/$(printf %05d $e).pth --ae data/result/$ae/0/model/00100.pth > $L/${ae}_eval.log 2>&1
    echo "$ae eval exit $? $(date)" >> $L/done_x
  done
  until [ -f data/result/v15_s1_${run}_l1/done_seed0 ]; do sleep 300; done
  python - $exp v15_s1_${run}_l1 >> $L/done_x 2>&1 <<'PY'
import sys, torch
a = torch.load(f'data/result/{sys.argv[1]}/0/model/00200.pth', map_location='cpu')['model']
b = torch.load(f'data/result/{sys.argv[2]}/0/model/00200.pth', map_location='cpu')['model']
ks = [k for k in a if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module', 'share_lns'))]
print(sys.argv[2], 'encoders identical to mse ep200:', all(torch.equal(a[k], b[k]) for k in ks), len(ks))
PY
  wait
}
one colourl1 $1 &
one colourl3 $2 &
one plain6k $3 &
wait
echo ALL_DONE >> $L/done_x
