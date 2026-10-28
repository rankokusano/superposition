#!/bin/bash
# v15 Sec.42 (Kusano-san 2026-10-29): delete the stage-1 checkpoints of epochs not used for any judgement, listed in
# docs/v15_stage1_ckpt_delete_list.txt (1690 files, 4.16 GB). Kept (docs/v15_stage1_ckpt_keep_list.txt, 35 files): mse ep100/150/200
# (v15 (i) Encoder = v15_s1_plain_mse ep200), l1 ep200, Fig.5 Decoder ep100. Refuses any path that is not a v15_s1_* checkpoint
# or is in the keep list. Run INSIDE the container (files are root-owned).
set -u
cd "$(dirname "$0")"
LIST=docs/v15_stage1_ckpt_delete_list.txt; KEEP=docs/v15_stage1_ckpt_keep_list.txt
bad=$(grep -v -E '^data/result/v15_s1_[a-z0-9_]+/0/model/[0-9]{5}\.pth$' $LIST | wc -l)
clash=$(grep -F -x -f $KEEP $LIST | wc -l)
[ $bad = 0 ] && [ $clash = 0 ] || { echo "refused: bad $bad clash $clash"; exit 1; }
n=0; while read -r f; do [ -f "$f" ] && rm -- "$f" && n=$((n+1)); done < $LIST
echo "deleted $n files $(date)"; missing=$(while read -r f; do [ -f "$f" ] || echo x; done < $KEEP | wc -l); echo "keep files missing: $missing"
