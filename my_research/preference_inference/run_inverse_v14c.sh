#!/bin/bash
# v14-A on the v14-C models (docs/v14_experiment_log.md Sec.5, 13, 21, 24, 25; exploratory per Sec.33).
# A-1 Red (main; no a1_liked_t = sign +1) and Green (--a1_liked green, Sec.25 projection check), l1 late-5 + ep200
# again, mse late-5 for the FP secondary, then the judgements / comparisons, FP direction and projection.
# Run INSIDE the container:  ALLOW_GPU=<gpu> bash run_inverse_v14c.sh <gpu>
set -u
cd "$(dirname "$0")"
gpu=$1
case $gpu in 5|6) ;; *) [ "${ALLOW_GPU:-}" = "$gpu" ] || { echo "GPU $gpu is not allowed for v14"; exit 1; } ;; esac
L=data/result/v14/logs
setsid nohup bash -c "
    export CUDA_VISIBLE_DEVICES=${gpu}
    E='python -u analyze/v14/eval_inverse_v14.py --out inverse'
    C='v14c:mixedpref:0 v14c:mixedpref:1 v14c:mixedpref:2 v14c:randompref:0 v14c:randompref:1'
    CM='v14cmse:mixedpref:0 v14cmse:mixedpref:1 v14cmse:mixedpref:2 v14cmse:randompref:0 v14cmse:randompref:1 v12mse:mixed:0 v12mse:mixed:1 v12mse:mixed:2'
    G12='v12:mixed:0 v12:mixed:1 v12:mixed:2 v12:random:0 v12:random:1 v12:random:2'
    G12M='v12mse:mixed:0 v12mse:mixed:1 v12mse:mixed:2 v12mse:random:0 v12mse:random:1 v12mse:random:2'
    \$E --models \$C > $L/inv14c_red.log 2>&1 &&
    \$E --models \$C --epochs 200 --tag_suffix _rep > $L/inv14c_red_rep.log 2>&1 &&
    \$E --models \$CM > $L/inv14c_mse.log 2>&1 &&
    \$E --models \$CM --epochs 200 --tag_suffix _rep > $L/inv14c_mse_rep.log 2>&1 &&
    \$E --models \$C \$G12 --a1_liked green > $L/inv14c_green.log 2>&1 &&
    \$E --models \$CM \$G12M --a1_liked green > $L/inv14c_green_mse.log 2>&1 &&
    cd analyze/v14 &&
    A='python -u aggregate_inverse_v14.py --sub inverse' &&
    \$A --pairs v14c:mixedpref:0,v12:mixed:0 v14c:mixedpref:1,v12:mixed:1 v14c:mixedpref:2,v12:mixed:2 --name aggregate_v14c_vs_v12exp > ../../$L/inv14c_agg.log 2>&1 &&
    \$A --pairs v14c:mixedpref:0,v12:random:0 v14c:mixedpref:1,v12:random:1 v14c:mixedpref:2,v12:random:2 --name aggregate_v14c_vs_v12noexp >> ../../$L/inv14c_agg.log 2>&1 &&
    \$A --pairs v14c:mixedpref:0,v14c:randompref:0 v14c:mixedpref:1,v14c:randompref:1 --name aggregate_v14c_vs_noexpv >> ../../$L/inv14c_agg.log 2>&1 &&
    \$A --pairs v14cmse:mixedpref:0,v12mse:mixed:0 v14cmse:mixedpref:1,v12mse:mixed:1 v14cmse:mixedpref:2,v12mse:mixed:2 --name aggregate_v14cmse_vs_v12exp >> ../../$L/inv14c_agg.log 2>&1 &&
    \$A --pairs v14cmse:mixedpref:0,v12mse:random:0 v14cmse:mixedpref:1,v12mse:random:1 v14cmse:mixedpref:2,v12mse:random:2 --name aggregate_v14cmse_vs_v12noexp >> ../../$L/inv14c_agg.log 2>&1 &&
    python -u fp_direction_v14.py --tags v14cmse_mixedpref_0 v14cmse_mixedpref_1 v14cmse_mixedpref_2 v14cmse_randompref_0 v14cmse_randompref_1 v12mse_mixed_0 v12mse_mixed_1 v12mse_mixed_2 --name fp_direction_v14cmse > ../../$L/inv14c_dir.log 2>&1 &&
    python -u fp_direction_v14.py --tags v14c_mixedpref_0 v14c_mixedpref_1 v14c_mixedpref_2 v14c_randompref_0 v14c_randompref_1 --epochs 200 --name fp_direction_v14cl1 >> ../../$L/inv14c_dir.log 2>&1 &&
    python -u projection_v14.py --tags v14c_mixedpref_0 v14c_mixedpref_1 v14c_mixedpref_2 v14c_randompref_0 v14c_randompref_1 v12_mixed_0 v12_mixed_1 v12_mixed_2 v12_random_0 v12_random_1 v12_random_2 --name projection_l1 > ../../$L/inv14c_proj.log 2>&1 &&
    python -u projection_v14.py --tags v14cmse_mixedpref_0 v14cmse_mixedpref_1 v14cmse_mixedpref_2 v14cmse_randompref_0 v14cmse_randompref_1 v12mse_mixed_0 v12mse_mixed_1 v12mse_mixed_2 v12mse_random_0 v12mse_random_1 v12mse_random_2 --name projection_mse >> ../../$L/inv14c_proj.log 2>&1
    echo \"exit \$?\" > ../../$L/inv14c.done
" < /dev/null > /dev/null 2>&1 &
echo "v14-A on v14-C chain on GPU ${gpu}"
