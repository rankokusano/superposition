#!/bin/bash
# fig5_v6_base: same procedure as fig5_v5_base (logs/fig5_v5_chain2.sh + logs/fig5c_recon.sh),
# pretrain = v6_s3_base_l1 ep200. Writes only under data/result/fig5_v6_base/.
set -u
cd /work/my_research/preference_inference
EXP=fig5_v6_base
DEV=cuda:0

echo "[fig5v6] $(date -u) train Autoencoder $EXP 100ep"
python train.py --exp_config $EXP --seed 0 --device $DEV > logs/fig5_v6_train.log 2>&1
echo "[fig5v6] train rc=$? last=$(ls data/result/$EXP/0/model/ 2>/dev/null | tail -1)"
[ -f data/result/$EXP/0/model/00100.pth ] || { echo "[fig5v6] no ep100 -- abort"; exit 1; }

echo "[fig5v6] verify encoder in saved checkpoints (frozen -> must equal v6 S3/S2)"
python analyze/verify_fig5_v6_encoder_load.py --saved_exp $EXP --epochs 0 10 100 > logs/fig5_v6_verify_saved.log 2>&1
rc=$?; tail -12 logs/fig5_v6_verify_saved.log
[ $rc -eq 0 ] || { echo "[fig5v6] encoder verification FAILED -- abort"; exit 1; }

df -h /work | tail -1
for E in 10 100; do
  echo "[fig5v6] $(date -u) grid test ep$E"
  python test.py --exp_config $EXP --seed 0 --device $DEV \
    --test_epoch $E --test_data_name grid --test_batch_size 100 --test_modes eval \
    --save_targets self_vision other_vision self_position other_position \
    --test_name grid_recon > logs/fig5_v6_test_ep${E}.log 2>&1
  echo "[fig5v6] test ep$E rc=$?"
done

d=data/result/$EXP/0/test/grid_recon/save
ln -sf /work/my_research/preference_inference/data/data/grid/data.h5 "$d/data.h5"
ln -sf /work/my_research/preference_inference/analyze "$d/analyze"
for E in 10 100; do
  ( cd "$d" && python analyze/analyze_vpt.py --epoch $E --margin 1 ) > logs/fig5_v6_vpt_ep${E}.log 2>&1
  echo "[fig5v6] vpt ep$E rc=$?"
  # vpt/1/ is keyed by margin, not epoch: keep a per-epoch copy before the next run overwrites it
  cp "$d/vpt/1/histogram/result.txt" "$d/vpt/1/histogram/result_ep${E}.txt"
  echo "[fig5v6] === ep$E result ==="; cat "$d/vpt/1/histogram/result_ep${E}.txt"
done
ls -la "$d/saved.h5"
df -h /work | tail -1
echo "[fig5v6] DONE $(date -u)"
