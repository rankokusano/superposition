#!/bin/bash
# Paper extra B4 (log Sec.29). Run INSIDE the container.
#   bash run_b4.sh data            : the two new test sets on CPU (nice 10), then B4/logs/data.done
#   GPU=<g> bash run_b4.sh <seed>  : G5b and tracking first; when data.done exists, G5a (G5b again) and the flipped G5a
set -u
cd "$(dirname "$0")/../.."
L=data/result/paper_extra/B4/logs; mkdir -p $L
if [ "$1" = data ]; then
    for d in paper_s2t_green_a1p80 paper_s2_pref_a1p80; do
        nice -n 10 xvfb-run --auto-servernum python -u simulation/collect_data_paper_b4.py --dataset $d > $L/data_$d.log 2>&1
    done
    echo "tracebacks $(grep -l Traceback $L/data_*.log | wc -l)" > $L/data.done
    exit 0
fi
s=$1; export CUDA_VISIBLE_DEVICES=${GPU:?}
R="python -u analyze/paper_extra/b4_run.py"
M="mixed80:$s signrand:$s random:$s"
$R g5b --models $M > $L/g5b_s$s.log 2>&1
$R track --models $M > $L/track_s$s.log 2>&1
$R track --models $M --epochs 200 --common > $L/track_common_s$s.log 2>&1
until [ -f $L/data.done ]; do sleep 30; done
for c in mixed80 signrand random; do
    for e in 160 170 180 190 200; do
        $R g5 --exp_fmt $(python -c "import sys; sys.path.insert(0, 'analyze/paper_extra'); import b4_run; print(b4_run.L1['$c'].format(s=$s))") \
            --conds $c --seed $s --epoch $e --sub ../paper_extra/B4/g5/$c/s$s > $L/g5_${c}_s${s}_e$e.log 2>&1
    done
done
sleep $((s * 60))    # decomp writes one shared reference file on its first run
$R decomp --seed $s --conds mixed80 signrand random > $L/decomp_s$s.log 2>&1
echo "tracebacks $(grep -l Traceback $L/*_s$s*.log | wc -l)" > $L/eval_s$s.done
