#!/bin/bash
# Paper round 13 rev2 Sec.1.6 (docs/paper_extra_log.md Sec.37.3; exploratory): matching candidates c+- (eval_inverse_v14,
# unchanged, via vdir_c_run.py) on E1's 50% sign-randomised runs, stage-A preference-mixed set (pref), both stages,
# late 5 (ep160-200), pairs 1-6; then the ep200 mse checkpoint of pair 1 again (_rep, bit-identity check). GPU1.
# Run INSIDE the container, detached:  GPU=1 setsid nohup bash analyze/paper_extra/run_vdir_c.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=${GPU:?}
[ "$GPU" = "0" ] && { echo "GPU0 is not to be used"; exit 1; }
L=data/result/paper_extra/Vdir_B/c_sr50/logs; mkdir -p $L
R="python -u analyze/paper_extra/vdir_c_run.py --sets pref --out ../paper_extra/Vdir_B/c_sr50/inverse"
for s in 0 1 2 3 4 5; do
    $R --models csr50mse:signrand50:$s > $L/inverse_mse_s$s.log 2>&1
    $R --models csr50:signrand50:$s > $L/inverse_l1_s$s.log 2>&1
done
$R --models csr50mse:signrand50:0 --epochs 200 --tag_suffix _rep > $L/inverse_mse_rep_s0.log 2>&1
echo "tracebacks $(grep -l Traceback $L/*.log | wc -l) $(date '+%m-%d %H:%M')" > $L/all.done
