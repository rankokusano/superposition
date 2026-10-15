#!/bin/bash
# v10 B1: run b1_checks_v10.py for each trained agent once its training has saved.
cd /work/my_research/preference_inference
L=data/result/v10_swap/b1/logs
g=${GPU0:-1}
for ag in a1 a2; do for s in ${SEEDS:-0 1 2}; do
  ( until grep -q '^Saved:.*critic' $L/train_${ag}_seed$s.log 2>/dev/null || grep -q Traceback $L/train_${ag}_seed$s.log; do sleep 30; done
    CUDA_VISIBLE_DEVICES=$g MESA_GL_VERSION_OVERRIDE=3.3 xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24' \
      python analyze/v10/b1_checks_v10.py --agent $ag --seed $s > $L/checks_${ag}_seed$s.log 2>&1
    echo "CHECKS_DONE ${ag}_seed$s" >> $L/checks_done.log ) &
  g=$((g+1))
done; done
wait
echo ALL_CHECKS_DONE ${SEEDS:-0 1 2} >> $L/checks_done.log
