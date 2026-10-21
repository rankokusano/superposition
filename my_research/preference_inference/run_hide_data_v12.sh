#!/bin/bash
# v12 proposal C data (docs/v12_experiment_log.md Sec.45): A-2-hidden input views, CPU only (no GPU). Detached, inside the container.
set -u
cd "$(dirname "$0")"
export MESA_GL_VERSION_OVERRIDE=3.3
L=data/data/v12_hide_logs; mkdir -p $L
NP=6
for src in v11_s2t_green v11_s2t_red v11_s2t_green_swap v11_s2_green v11_s2_pref; do
  splits=$(python -c "import h5py; print(' '.join(h5py.File('data/data/$src/data.h5','r').keys()))")
  for sp in $splits; do
    for p in $(seq 0 $((NP-1))); do
      xvfb-run --auto-servernum --server-args="-screen 0 1024x768x24" \
        python analyze/v12/make_hide_a2_v12.py --src $src --split $sp --part $p --nparts $NP > $L/${src}_${sp}_$p.log 2>&1 &
    done
    wait
  done
  python analyze/v12/make_hide_a2_v12.py --src $src --merge --nparts $NP > $L/${src}_merge.log 2>&1 && rm -rf data/data/v12_hide_$src/parts
  echo "$src exit $? $(date)" >> $L/done
done
echo "all done $(date)" >> $L/done
