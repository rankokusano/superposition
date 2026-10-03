#!/bin/bash
# P0-1 unified evaluation harness (research_plan_v4_v5.md Sec.P0-1).
# Runs the CANONICAL protocol (report #1): dataset = r2_a1random_a2rl,
# split = the `eval` group (= training split, unshuffled -- same group the
# original paper's Fig.4d regression uses, see v4_experiment_log Sec.7.14),
# mask ON (config p_mask_vision), --eval_seed fixed, --cudnn_deterministic.
#
# Usage (from /work/my_research/preference_inference, inside the container):
#   bash analyze/run_eval_v4v5.sh <exp_config> <epoch> [eval_seed] [dataset] [train_seed]
# e.g.
#   bash analyze/run_eval_v4v5.sh r4_ve_exp3parity 200 0
#   bash analyze/run_eval_v4v5.sh r3_a_direct_1000pretrain 200 0 r2_a1random_a2rl 1   # seed-1 checkpoint
#
# For report #2 (in-distribution) pass the model's own training dataset as
# <dataset>; for report #3 (held-out) additionally pass --test_modes test
# (edit MODE below) -- keep those as separate labelled runs.
set -e

EXP="${1:?exp_config}"
EPOCH="${2:?epoch}"
ESEED="${3:-0}"
DATASET="${4:-r2_a1random_a2rl}"
MODE="eval"                       # canonical: eval group (= training split)
SEED="${5:-0}"                    # training seed of the checkpoint (which <exp_config>/<seed>/ to load)
BATCH=10
TESTNAME="${DATASET}_canon_es${ESEED}"
LABEL="${EXP}_s${SEED}_ep${EPOCH}_${DATASET}_es${ESEED}"

RESULT_DIR="data/result/${EXP}/${SEED}"
SAVED_H5="${RESULT_DIR}/test/${TESTNAME}/save/saved.h5"

echo "=== [1/5] test.py -> saved.h5  (${EXP} s${SEED} ep${EPOCH}, ${DATASET}/${MODE}, eval_seed=${ESEED}) ==="
python -u test.py \
    --exp_config "${EXP}" --seed "${SEED}" --device "${DEVICE:-cuda:0}" \
    --test_epoch "${EPOCH}" --test_data_name "${DATASET}" --test_name "${TESTNAME}" \
    --test_batch_size "${BATCH}" --test_modes "${MODE}" \
    --eval_seed "${ESEED}" --cudnn_deterministic \
    --save_targets self_position other_position state q2_hat r_hat a2_target_landmark

echo "=== [2/5] regression_baseline_v4.py -> 4-axis R^2 ==="
python analyze/regression_baseline_v4.py \
    --saved_h5 "${SAVED_H5}" --epoch "${EPOCH}" --mode "${MODE}" \
    --label "${LABEL}" --exp_config "${EXP}" --seed "${SEED}" --dataset_name "${DATASET}"

echo "=== [3/5] probe_q_direct_regression.py (single-timestep Q -> position) ==="
python analyze/probe_q_direct_regression.py --exp_config "${EXP}" --epoch "${EPOCH}" --seed "${SEED}" \
    --eval_seed "${ESEED}" --label "${LABEL}_probeq" || echo "(skipped: not a ProbeQ model or no probe_actions)"

if python -c "import yaml,sys; c=yaml.safe_load(open('config/exp/${EXP}.yml')); n=c['model']['name']; sys.exit(0 if ('ValueEstimation' in n and 'V6' not in n) else 1)"; then
  echo "=== [4/5] r4_ve_eval.py (real/zero/true_q2/constant_mean + Q2hat stats + direction) ==="
  python analyze/r4_ve_eval.py --exp_config "${EXP}" --epoch "${EPOCH}" --seed "${SEED}" \
      --eval_seed "${ESEED}" --label "${LABEL}_r4ve"
  echo "=== [5/5] q2_position_regression.py (Q2hat -> A-1 pos vs A-2 pos) ==="
  python analyze/q2_position_regression.py --exp_config "${EXP}" --epoch "${EPOCH}" --seed "${SEED}" \
      --eval_seed "${ESEED}" --label "${LABEL}_q2pos"
else
  echo "=== [4-5/5] skipped (model has no Value Estimator) ==="
fi

echo "=== training curve ==="
python analyze/plot_training_curve.py --exp_config "${EXP}" || true

echo "=== DONE: ${LABEL}  ->  data/result/baseline_v4/${LABEL}_*.{json,txt}  +  ${SAVED_H5} ==="
