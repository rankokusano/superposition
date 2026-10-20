#!/bin/bash
# v12 shaped stage-2 dry run (docs/v12_experiment_log.md Sec.34): 2-epoch trainings from (P-c) ep150 for both shapes
# (mask on GPU3/4, signed_pairs on GPU5/6), a check that Q2 has the restricted form, then the stage-2 evaluation
# end to end (run_s2eval_v12.sh with epochs 1 2). Not a result. Run INSIDE the container, detached.
set -u
cd "$(dirname "$0")"
L=data/result/v12/trial/logs/s2shaped; mkdir -p $L
i=3
for sh in m s; do for st in green pref; do
  for c in random mixed; do
    exp=v12_test_s2${sh}_${st}_${c}_s0; mkdir -p data/result/$exp
    CUDA_VISIBLE_DEVICES=$i python -u train_v12.py --exp_config $exp --seed 0 --device cuda:0 --cudnn_deterministic --data_load_memory > data/result/$exp/train_seed0.log 2>&1 &
  done; i=$((i+1))
done; done
wait; echo "train done $(date)" >> $L/steps.log
CUDA_VISIBLE_DEVICES=3 python - > $L/shape_check.log 2>&1 <<'PY'
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'analyze/v12')
import h5py, numpy as np, torch, util
from eval_trial_v12 import load_v12
dev = torch.device('cuda:0')
g = h5py.File('data/data/v11_s2t_green/data.h5', 'r')['test']
x = torch.from_numpy(util.scale_vision(util.transpose_vision(g['self_vision'][:50]))).to(dev)
mo = torch.from_numpy(g['self_motion'][:50]).float().to(dev)
for sh in ('m', 's'):
    for st in ('green', 'pref'):
        for c in ('random', 'mixed'):
            m, _ = load_v12(f'v12_test_s2{sh}_{st}_{c}_s0', 0, 2, dev)
            m.init_state(50); Q = []
            with torch.no_grad():
                for t in range(100):
                    Q.append(m({'self_vision': x[:, t], 'self_motion': mo[:, t]}, 0 if t == 0 else 1, 0 if t == 0 else 1)['q2_hat'].cpu().numpy())
            Q = np.stack(Q, 1).reshape(-1, 8)
            r = {'bins2_5_max_abs': float(np.abs(Q[:, 2:6]).max()), 'max_abs': float(np.abs(Q).max()),
                 'std_per_bin': Q.std(0).round(3).tolist()}
            if sh == 's':
                nz = (np.abs(Q[:, [0, 1, 6, 7]]) > 1e-6).all(1)
                r['frac_bins01_same_sign'] = float((np.sign(Q[nz, 0]) == np.sign(Q[nz, 1])).mean())
                r['frac_01_vs_67_opposite'] = float((np.sign(Q[nz, 0]) == -np.sign(Q[nz, 6])).mean())
            print(sh, st, c, r)
PY
echo "shape_check exit $? $(date)" >> $L/steps.log
for sh in m s; do
  PREFIX=v12_test_s2$sh OUT=trial/s2shaped_$sh EPOCHS='1 2' WAIT=0 bash run_s2eval_v12.sh 0 $([ $sh = m ] && echo 3 || echo 5) &
done
wait
echo "all done $(date)" >> $L/steps.log
