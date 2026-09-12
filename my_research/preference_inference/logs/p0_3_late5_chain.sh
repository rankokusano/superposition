#!/bin/bash
# 3-seed late-5, contribution-1 error bars. BASE-stage (no VE) -> 4-axis R^2 only.
# CRITICAL: do NOT set CUDA_VISIBLE_DEVICES (some seeds pickled ckpts on cuda:5;
# torch.load has no map_location, so all GPUs must be visible). Pin compute to
# GPU7 with DEVICE=cuda:7.
set -u
cd /work/my_research/preference_inference
unset CUDA_VISIBLE_DEVICES
export DEVICE=cuda:7

do_run() {   # $1=exp  $2=E  $3=dataset  $4=seed
  local exp="$1" E="$2" ds="$3" s="$4"
  local log="logs/p0_3_${exp}_s${s}.log"
  echo "############## ${exp} seed${s} (${ds})  start $(date -u) ##############"
  bash analyze/run_eval_lateckpt.sh "$exp" "$E" "$s" "$ds" > "$log" 2>&1
  echo "  ${exp} seed${s} rc=$?  end $(date -u)"
  grep -E "h1_to_self|h2_to_self|mean \+/- sd|SURVIVES|FRAGILE" "$log" | tail -8
}

for s in 0 1 2; do do_run r3_a_direct_1000pretrain_400 400 r2_a1random_a2rl $s; done
for s in 0 1 2; do do_run r3_stay_400 400 r3_stay $s; done

echo "########## CONVERGENCE ##########"
python analyze/check_convergence.py r3_a_direct_1000pretrain_400 r3_stay_400 --seeds 0 1 2 2>&1 | grep -v "^saved:"
echo "[p0_3-late5] ALL DONE $(date -u)"
