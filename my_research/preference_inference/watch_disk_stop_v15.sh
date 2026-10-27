#!/bin/bash
# v15 Sec.30.9: if the free space on /work drops below 5 GB, place data/result/v15/STOP_NEW (no new (A)/(B) chain starts; running
# ones finish) and record the time. Checks every 2 minutes until ALL_DONE_AB. Run INSIDE the container, detached.
cd "$(dirname "$0")"
L=data/result/v15/stage2/logs
until grep -q ALL_DONE_AB $L/done_AB 2>/dev/null; do
  avail=$(df -BG --output=avail /work | tail -1 | tr -dc 0-9)
  if [ "$avail" -lt 5 ] && [ ! -f data/result/v15/STOP_NEW ]; then
    touch data/result/v15/STOP_NEW; echo "STOP_NEW placed: free ${avail} GB $(date)" >> $L/disk_watch
  fi
  sleep 120
done
