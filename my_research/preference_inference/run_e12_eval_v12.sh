#!/bin/bash
# v12 E1/E2 evaluation (docs/v12_experiment_log.md Sec.53). Waits for each checkpoint. Encoder measures on CPU;
# Fig.5 Decoders: E1 on GPU5, E2 on GPU6 (never GPU0; training is on GPU3/4). Run INSIDE the container, detached.
set -u
cd "$(dirname "$0")"
L=data/result/v12/e12/logs; mkdir -p $L
wait_ck() { until [ -f data/result/$1/0/model/$(printf %05d $2).pth ]; do sleep 120; done; sleep 30; }
one() { local t=$1 exp=$2 gpu=$3
  for e in 100 150 200; do
    wait_ck $exp $e
    CUDA_VISIBLE_DEVICES= python -u analyze/v12/eval_e12_v12.py --exp $exp --epochs $e > $L/${t}_enc_ep$e.log 2>&1
    echo "$t enc ep$e exit $? $(date)" >> $L/done
    CUDA_VISIBLE_DEVICES=$gpu python -u train_v12.py --exp_config v12_${t}_fig5_mse$e --seed 0 --device cuda:0 \
      --cudnn_deterministic --data_load_memory > $L/${t}_fig5_train_ep$e.log 2>&1
    CUDA_VISIBLE_DEVICES=$gpu python -u analyze/v11/fig5_v11.py --ae v12_${t}_fig5_mse$e --seed 0 --epoch 100 \
      --sub ../v12/e12/fig5_${t}_mse$e > $L/${t}_fig5_eval_ep$e.log 2>&1
    echo "$t fig5 ep$e exit $? $(date)" >> $L/done
  done
  if [ $t = e2 ]; then   # G2/G3 (own side) of the experienced run, mse late-5
    CUDA_VISIBLE_DEVICES=$gpu python -u analyze/v12/eval_trial_v12.py --exp_fmt v12_e2_full_{}_mse --conds mixed --seed 0 \
      --epochs 160 170 180 190 200 --no_p2 --out e12/s1eval_e2 > $L/e2_s1eval.log 2>&1
    echo "e2 s1eval exit $? $(date)" >> $L/done
  fi
  until [ -f data/result/${exp%_mse}_l1/done_seed0 ]; do sleep 300; done
  python - ${exp} ${exp%_mse}_l1 >> $L/done 2>&1 <<'PY'
import sys, torch
a = torch.load(f'data/result/{sys.argv[1]}/0/model/00200.pth', map_location='cpu')['model']
b = torch.load(f'data/result/{sys.argv[2]}/0/model/00200.pth', map_location='cpu')['model']
ks = [k for k in a if k.startswith(('self_vision_encoder_module', 'other_vision_encoder_module', 'share_lns'))]
print(sys.argv[2], 'encoders identical to mse ep200:', all(torch.equal(a[k], b[k]) for k in ks), len(ks))
PY
  CUDA_VISIBLE_DEVICES= python -u analyze/v12/eval_e12_v12.py --exp ${exp%_mse}_l1 --epochs 200 > $L/${t}_enc_l1.log 2>&1
  echo "$t enc l1 exit $? $(date)" >> $L/done
}
# comparison rows (same script): trial P-a and v12 trial body (encoders from scratch), ep150
(CUDA_VISIBLE_DEVICES= python -u analyze/v12/eval_e12_v12.py --exp v12_trial_pa_random_mse --epochs 150 > $L/ref_pa.log 2>&1
 CUDA_VISIBLE_DEVICES= python -u analyze/v12/eval_e12_v12.py --exp v12_trial_mixed_mse --epochs 150 > $L/ref_v12trial.log 2>&1
 echo "refs exit $? $(date)" >> $L/done) &
one e1 v12_e1_paper_random_mse 5 &
one e2 v12_e2_full_mixed_mse 6 &
wait
echo "all done $(date)" >> $L/done
