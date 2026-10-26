#!/bin/bash
# v15 Sec.24.5 data: A-2-hidden input views (train 1000t; test views for the generated / control pairs). CPU, nice 10.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES= MESA_GL_VERSION_OVERRIDE=3.3
L=logs/v15; mkdir -p $L
X="nice -n 10 xvfb-run --auto-servernum --server-args=-screen\ 0\ 1024x768x24 python analyze/v15/make_hide_a2_v15.py"
part() { nice -n 10 xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' python analyze/v15/make_hide_a2_v15.py "$@"; }
export -f part
{ for p in 0 1 2 3; do echo "--src v15_s2_c2_p80_half_1000t --mode train --part $p --nparts 4"; done
  for s in genx geny; do for p in 0 1; do echo "--src v15_s2_${s}_p80 --mode test --te_from v15_s2_genx_p80 --part $p --nparts 2"; done; done
  for s in c3x c3xp; do for p in 0 1; do echo "--src v15_s2_${s}_p80 --mode test --te_from v15_s2_c3x_p80 --part $p --nparts 2"; done; done
} | xargs -P 6 -I{} bash -c 'part {} >> logs/v15/hideA2_parts.log 2>&1; echo "parts {} exit $?"' 
python analyze/v15/make_hide_a2_v15.py --src v15_s2_c2_p80_half_1000t --mode train --merge --nparts 4 > $L/hideA2_merge_train.log 2>&1; echo "merge train exit $?"
for s in genx geny; do python analyze/v15/make_hide_a2_v15.py --src v15_s2_${s}_p80 --mode test --te_from v15_s2_genx_p80 --merge --nparts 2 > $L/hideA2_merge_$s.log 2>&1; echo "merge $s exit $?"; done
for s in c3x c3xp; do python analyze/v15/make_hide_a2_v15.py --src v15_s2_${s}_p80 --mode test --te_from v15_s2_c3x_p80 --merge --nparts 2 > $L/hideA2_merge_$s.log 2>&1; echo "merge $s exit $?"; done
nice -n 10 python analyze/v15/check_nonempty_v15.py v15_s2_c2_p80_half_1000t_hideA2 > $L/nonempty_hideA2.log 2>&1; echo "nonempty exit $?"
echo ALL_DONE
