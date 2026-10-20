#!/bin/bash
# v12 (docs/v12_experiment_log.md Sec.29): for pairs 2 then 1, wait for both l1 chains, then selection criterion (h2),
# G5 (+ D6-like secondary), and if the criterion is met launch the pair's stage-2 runs on the reported GPUs.
# Finally the stage-1 gate aggregation and figures. Run INSIDE the container from the v11 worktree's
# my_research/preference_inference. GPU0 never used.
set -u
cd "$(dirname "$0")"
L=data/result/v12/logs/eval_s1
declare -A S2G1=( [2]=3 [1]=7 ) ; declare -A S2G2=( [2]=4 [1]=7 )
for s in 2 1; do
  until [ -f data/result/v12_s1_random_l1_s$s/chain_seed$s.done ] && [ -f data/result/v12_s1_mixed_l1_s$s/chain_seed$s.done ]; do sleep 60; done
  echo "s$s chains: $(cat data/result/v12_s1_random_l1_s$s/chain_seed$s.done) / $(cat data/result/v12_s1_mixed_l1_s$s/chain_seed$s.done) at $(date)" >> $L/after_l1.log
  CUDA_VISIBLE_DEVICES=7 python analyze/v12/v11fig.py h2 --runs random:$s:200 mixed:$s:200 --sub ../v12/main/h2_check_s$s > $L/h2_s$s.log 2>&1
  ok=$(python - $s <<'PY'
import json, sys, glob
s = sys.argv[1]
f = sorted(glob.glob(f'data/result/v12/main/h2_check_s{s}/*.json'))[-1]
d = json.load(open(f))
vals = [d[k]['ridge_xy'] for k in (f'random:{s}:200', f'mixed:{s}:200')]
print('yes' if all(min(v) >= 0.8 for v in vals) else 'no', vals)
PY
)
  echo "s$s selection: $ok" >> $L/after_l1.log
  case "$ok" in yes*) bash run_stage2_v12.sh $s ${S2G1[$s]} ${S2G2[$s]} >> $L/after_l1.log 2>&1 ;; esac
  GPU=7 bash run_eval_s1_v12.sh G5 $s; echo "G5 s$s exit $?" >> $L/after_l1.log
done
until [ -f $L/FIG5.done ]; do sleep 60; done
GPU=7 bash run_eval_s1_v12.sh AGG; echo "AGG exit $?" >> $L/after_l1.log
echo done > $L/after_l1.done
