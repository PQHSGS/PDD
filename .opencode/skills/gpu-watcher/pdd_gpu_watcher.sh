#!/usr/bin/env bash
set -u
PID=3566536
VRAM_THRESHOLD_GB=15
WATCH_LOG=/tmp/opencode/pdd_watcher.log
RUN_LOG=/tmp/opencode/pdd_run.log
TRIGGER_LOG=/tmp/opencode/pdd_triggered.flag

log() { echo "$(date '+%F %T') $*" >> "$WATCH_LOG"; }

log "watcher started for pid=$PID threshold=${VRAM_THRESHOLD_GB}GiB"

while true; do
  pid_alive=no
  if kill -0 "$PID" 2>/dev/null; then pid_alive=yes; fi

  vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')

  if [ "$pid_alive" = "no" ]; then
    log "pid $PID gone -> triggering"
    break
  fi
  if [ -n "$vram_mib" ] && [ "$vram_mib" -lt $((VRAM_THRESHOLD_GB * 1024)) ]; then
    log "vram freed (${vram_mib}MiB < ${VRAM_THRESHOLD_GB}GiB) -> triggering"
    break
  fi
  sleep 15
done

log "launching in tmux 'pdd': python -u -m pdd.cli --config configs/qwen3_1.7b_base.json"

# ensure a pdd tmux session exists (keep the user's existing one if present)
tmux has-session -t pdd 2>/dev/null || tmux new-session -d -s pdd

# clear any leftover pane and start a fresh shell
tmux send-keys -t pdd "clear" Enter

# start the run inside tmux; pipe its output to the run log
tmux send-keys -t pdd "cd /mnt/disk4/pquan/PDD && source /mnt/disk1/miniconda3/etc/profile.d/conda.sh && conda activate pdd && python -u -m pdd.cli --config configs/qwen3_1.7b_base.json" Enter
tmux pipe-pane -t pdd -o "cat >> $RUN_LOG"
log "command sent to tmux 'pdd'; run log at $RUN_LOG"
