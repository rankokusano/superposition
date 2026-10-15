#!/bin/bash
# v10 stage A: late-5 evaluation of one (exp_config, seed) on A0 (r2_a1random_a2rl)
# and A1 (v10_swap_r2). Outputs only under data/result/v10_swap/stageA/eval/.
# Each saved.h5 is deleted only after the per-checkpoint JSON exists.
#
# Usage (inside the container, from /work/my_research/preference_inference):
#   bash analyze/v10/run_stageA_eval.sh <exp_config> <seed> <device> "<epochs>"
#   e.g. bash analyze/v10/run_stageA_eval.sh r3_a_direct_1000pretrain_400 0 cuda:3 "360 370 380 390 400"
set -e
EXP="${1:?exp_config}"; SEED="${2:?seed}"; DEV="${3:?device}"; EPOCHS="${4:?epochs}"
ROOT="data/result/v10_swap/stageA/eval/${EXP}/s${SEED}"

for E in ${EPOCHS}; do
  JSON="${ROOT}/ep${E}/stageA_regression.json"
  if [ -f "${JSON}" ]; then echo "skip ep${E} (json exists)"; continue; fi
  for C in A0 A1; do
    if [ "$C" = A0 ]; then DS=r2_a1random_a2rl; else DS=v10_swap_r2; fi
    OUT="${ROOT}/ep${E}/${C}"
    if [ ! -f "${OUT}/save/saved.h5" ]; then
      echo "=== ${EXP} s${SEED} ep${E} ${C} (${DS}) ==="
      python -u analyze/v10/test_v10.py --exp_config "${EXP}" --seed "${SEED}" --device "${DEV}" \
        --test_epoch "${E}" --test_data_name "${DS}" --test_batch_size 10 --test_modes eval test \
        --eval_seed 0 --cudnn_deterministic --save_targets state self_position other_position \
        --out_dir "${OUT}" 2>&1 | grep -v 'it/s\]' | grep -v -i warn
    fi
  done
  python analyze/v10/stageA_regression.py --a0_dir "${ROOT}/ep${E}/A0" --a1_dir "${ROOT}/ep${E}/A1" \
    --epoch "${E}" --exp_config "${EXP}" --seed "${SEED}" --out "${JSON}" > /dev/null
  if [ -f "${JSON}" ]; then
    rm -f "${ROOT}/ep${E}/A0/save/saved.h5" "${ROOT}/ep${E}/A1/save/saved.h5"
    echo "done ep${E}: ${JSON} (saved.h5 removed)"
  fi
done
