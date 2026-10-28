#!/bin/bash
# Paper round 5 work 1 (B7, log Sec.31). Run INSIDE the container: GPU=<g> bash run_b7.sh <seed>
set -u
cd "$(dirname "$0")/../.."
s=$1; export CUDA_VISIBLE_DEVICES=${GPU:?}
L=data/result/paper_extra/B7/logs; mkdir -p $L
M="pref:$s nonesigned:$s mixed50:$s"; [ $s -lt 2 ] && M="$M noexpv:$s"
python -u analyze/paper_extra/b7_dp.py --models $M > $L/b7_s$s.log 2>&1
python -u analyze/paper_extra/b7_dp.py --models $M --epochs 200 --tag _rep > $L/b7_rep_s$s.log 2>&1
echo "tracebacks $(grep -l Traceback $L/*_s$s.log | wc -l)" > $L/b7_s$s.done
