#!/bin/bash
# v7 remake 1 (docs/v7_experiment_log.md §2.13): train, then run every
# pre-registered post-training step and stop where a judgement is needed.
# Run inside Docker:  bash /work/my_research/preference_inference/simulation/run_retrain_r1_v7.sh <git_commit>
set -u
PI=/work/my_research/preference_inference
COMMIT=${1:-unknown}
R=data/result/v7_irl/retrain
ENV_CFG=/work/simulation/config/collect/self_random_other_stay.yml
XV="xvfb-run --auto-servernum"

collapsed() {  # $1 = health txt; exit 0 if collapsed
  python - "$1" <<'EOF'
import re, sys
s = [float(x) for x in re.findall(r'std across corners: ([0-9.]+)', open(sys.argv[1]).read())]
sys.exit(0 if len(s) == 3 and all(v < 0.001 for v in s) else 1)
EOF
}

job() {  # gpu seed sampling episodes tag mode(full|critic)
  local GPU=$1 SEED=$2 SAMP=$3 EPS=$4 TAG=$5 MODE=$6 NAME="seed$2_$5"
  cd $PI || exit 1
  CUDA_VISIBLE_DEVICES=$GPU $XV python -u simulation/train_rl_v7.py --seed $SEED --sampling $SAMP \
      --episodes $EPS --tag $TAG --alpha_min 0.002 > logs/v7_train_${TAG}_seed${SEED}.log 2>&1 \
      || { echo "[$NAME] TRAIN FAILED"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU $XV python analyze/check_critic_health_v6.py \
      --critic_path data/model/v7/v7_rl_critic_${NAME}.pth --env_config $ENV_CFG --camera self --scan_self \
      --film --label v7_${NAME} > $R/health_${NAME}.txt 2>&1
  if collapsed $R/health_${NAME}.txt; then echo "[$NAME] COLLAPSED -- stopping this job"; return 0; fi
  if [ "$MODE" = full ]; then
    ARGS="--rollout --check1 --check2 --actor_path data/model/v7/v7_rl_actor_${NAME}.pth --rollout_key v7_retrain_${NAME}"
  else
    ARGS="--check2"
  fi
  CUDA_VISIBLE_DEVICES=$GPU $XV python -u analyze/feasibility_goals_v7.py $ARGS \
      --critic_path data/model/v7/v7_rl_critic_${NAME}.pth --out_dir $R/${NAME} --git_commit $COMMIT \
      > logs/v7_retrain_checks_${NAME}.log 2>&1 || { echo "[$NAME] CHECKS FAILED"; return 1; }
  echo "[$NAME] done"
}

job 1 3 named_mixheavy 3500 condA_r1 full &
job 2 4 named_mixheavy 3500 condA_r1 full &
job 3 2 v6 1000 condB_r1 critic &
wait

cd $PI
PAIRS="v6_seed2=$R/v6_seed2_ref"
for N in seed3_condA_r1 seed4_condA_r1 seed2_condB_r1; do
  [ -f $R/$N/check2_maps.npz ] && PAIRS="$PAIRS $N=$R/$N"
done
VF_OUT=$R/vector_field_criterion_r1.json python analyze/vector_field_criterion_v7.py $PAIRS > logs/v7_retrain_r1_vector_field.log 2>&1
CMP="v6 seed2 (ref)=$R/v6_seed2_ref"
for N in seed3_condA_r1 seed4_condA_r1 seed2_condB_r1; do
  [ -f $R/$N/check2_maps.npz ] && CMP="$CMP|$N=$R/$N"
done
IFS='|' read -ra CMPA <<< "$CMP"
python analyze/feasibility_goals_v7.py --out_dir $R/r1 --compare "${CMPA[@]}" > logs/v7_retrain_r1_compare.log 2>&1
python analyze/retrain_summary_v7.py --round r1 --critic_seed 3 --actor_seed 4 --b_seed 2 > logs/v7_retrain_r1_summary.log 2>&1
echo "PIPELINE DONE -- stopping for judgement (no stage-0 recollection is started)"
