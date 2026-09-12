#!/bin/bash
# P1.75: wait for symmetric data collection to finish, verify the h5, then
# launch r4_move_sym training on GPU7. research_plan_v4_v5.md Sec.4ter.
set -u
cd /work/my_research/preference_inference
CLOG=logs/p1_75_collect_r2_a1red_a2red.log
TLOG=logs/p1_75_train_r4_move_sym.log
H5=data/data/r2_a1red_a2red/data.h5

running() {
  for p in /proc/[0-9]*; do
    [ -r "$p/cmdline" ] || continue
    tr '\0' ' ' < "$p/cmdline" 2>/dev/null | grep -q collect_data_r2_symmetric && return 0
  done
  return 1
}

echo "[chain] waiting for collection to finish ..."
while running; do sleep 60; done
echo "[chain] collection process gone. verifying h5 ..."

python - <<'PY'
import h5py, sys
try:
    f = h5py.File('data/data/r2_a1red_a2red/data.h5', 'r')
except Exception as e:
    print('[chain] h5 open FAILED:', e); sys.exit(1)
need = ['self_vision', 'other_vision', 'self_position', 'other_position']
for split in ('train', 'test'):
    if split not in f:
        print('[chain] missing split', split); sys.exit(1)
    for k in need:
        if k not in f[split]:
            print(f'[chain] missing {split}/{k}'); sys.exit(1)
print('[chain] h5 OK  train/self_vision', f['train/self_vision'].shape,
      ' test/self_vision', f['test/self_vision'].shape,
      ' attrs:', dict(f.attrs))
f.close()
PY
if [ $? -ne 0 ]; then
  echo "[chain] h5 verification failed -- NOT starting training"
  exit 1
fi

echo "[chain] tail of collection log:"
tail -n 5 "$CLOG"

echo "[chain] launching r4_move_sym training on GPU7 -> $TLOG"
export CUDA_VISIBLE_DEVICES=7
setsid nohup python train.py --exp_config r4_move_sym --seed 0 --device cuda:0 \
  > "$TLOG" 2>&1 < /dev/null &
echo "[chain] train pid $!"
