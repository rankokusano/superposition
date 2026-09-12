#!/bin/bash
set -u; cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES; export DEVICE=cuda:6
gen() { # exp epoch
  echo "[viz] $(date -u) $1 ep$2 on r3_stay_viz20"
  python test.py --exp_config "$1" --seed 0 --device cuda:6 \
    --test_epoch "$2" --test_data_name r3_stay_viz20 --test_batch_size 10 --test_modes eval \
    --save_targets self_vision other_vision self_position other_position \
    --test_name viz20 --eval_seed 0 --cudnn_deterministic > "logs/viz_${1}.log" 2>&1
  echo "[viz] $1 rc=$?"
  python3 -c "
import h5py
f=h5py.File(\"data/result/$1/0/test/viz20/save/saved.h5\")
g=list(f.keys())[-1]
ks=[]
f[g].visit(lambda n: ks.append(n))
print(\"  \", [k for k in ks if k.endswith((\"input\",\"truth\",\"prediction\"))])
"
}
gen v5_base_l1 200
gen r3_stay_400 400
gen exp1_l1_1000 200
echo "[viz] DONE $(date -u)"
