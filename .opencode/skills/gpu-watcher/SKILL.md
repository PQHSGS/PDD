---
name: gpu-watcher
description: Check what a PID/process is doing, monitor its progress, and build a detached watcher that triggers a command when a process exits or GPU VRAM frees. Activates on requests like "check this pid", "monitor this process", "run X when that job finishes / when vram drops", or intent to chain a GPU job after another.
---

# GPU / Process Watcher

The trick for "check on someone else's running job, then chain my own GPU job
the moment theirs finishes." Built and battle-tested in the PDD repo.

## 1. Inspect a PID

Always use `rtk`-style wrappers (agent env has `rtk ps`, `rtk cat`, `rtk read`,
`rtk ls`, `rtk grep`) to avoid shell redirection surprises.

```bash
# what is it + who owns it
rtk ps -p <PID> -o pid,ppid,user,etime,cmd

# working directory + full command line + open fds (see GPU/devices)
rtk ls -l /proc/<PID>/cwd
tr '\0' ' ' < /proc/<PID>/cmdline; echo
rtk ls -l /proc/<PID>/fd 2>/dev/null | tail -5
```

`/proc/<PID>/fd` shows `-> /dev/nvidiaN` to learn which GPU it's on — never kill
an active process on the 4090.

## 2. Check progress

Look for the output dir in the command line (`--output-dir`), then:

```bash
rtk ls -la <output-dir>/            # checkpoints: epoch_100.pth, last.pth, ...
rtk tail -5 <output-dir>/metrics.jsonl   # epochs, loss, LR, seconds/epoch
```

Estimate ETA from `epoch / total_epochs` and `seconds` per epoch.

## 3. The watcher

A detached poller that triggers a command when **either** condition fires:

- the tracked **PID exits** (`kill -0`), or
- **VRAM on GPU 0 drops below a threshold** (`nvidia-smi`), whichever comes first.

Then it launches your command **inside a tmux session** so you can attach and
watch it live.

Script lives at `/tmp/opencode/pdd_gpu_watcher.sh` (template below). Logs:

| Path | Contents |
|---|---|
| `WATCH_LOG` `/tmp/opencode/pdd_watcher.log` | watcher status + trigger line |
| `RUN_LOG` `/tmp/opencode/pdd_run.log` | piped pane output of the launched command |
| `TRIGGER_LOG` `/tmp/opencode/pdd_triggered.flag` | `rc=` written when command exits |

### Template

```bash
#!/usr/bin/env bash
set -u
PID=<TRACKED_PID>
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

# launch inside a tmux session so the user can attach and watch
log "launching in tmux '<SESSION>': <COMMAND>"
tmux has-session -t <SESSION> 2>/dev/null || tmux new-session -d -s <SESSION>
tmux send-keys -t <SESSION> "clear" Enter
tmux send-keys -t <SESSION> "<ENV_SETUP && COMMAND>" Enter
tmux pipe-pane -t <SESSION> -o "cat >> $RUN_LOG"
log "command sent to tmux '<SESSION>'; run log at $RUN_LOG"
```

### Launch it correctly (CRITICAL)

```bash
setsid /tmp/opencode/pdd_gpu_watcher.sh >/dev/null 2>&1 < /dev/null & disown
```

**Why `setsid` is mandatory:** plain `nohup … & disown` still lives in the
tool's process group. When a shell/tool command times out or gets aborted, the
shell tears down its whole process group — the watcher dies silently (SIGHUP is
only one hazard; process-group kill is the real killer). `setsid` detaches it
into its own session so it survives every subsequent tool call. Verify after
launch:

```bash
pgrep -f pdd_gpu_watcher.sh -l   # expect exactly one live bash process
rtk tail -1 /tmp/opencode/pdd_watcher.log   # "watcher started …"
```

If two PIDs match, one was the transient shell — re-run `pgrep -l` to confirm
only one stable `bash` remains.

### Kill / change threshold

```bash
pkill -f pdd_gpu_watcher.sh      # stop old watcher
# edit VRAM_THRESHOLD_GB in the script, then re-launch with setsid (above)
```

### Trigger conditions & semantics

- Poll interval `sleep 15` → trigger fires at most 15s late. Fine for chaining jobs.
- PID check first, then VRAM check; either one breaking the loop triggers.
- The tmux command is sent via `send-keys`, not run as a child — so the watcher
  finishes after sending; `TRIGGER_LOG` reflects the launched job's `rc` only if
  you keep the tmux session alive with `pipe-pane` and wait (adjust as needed).

## 4. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `pgrep` shows nothing after launch | launch command hung/timed out and its process group was reaped → relaunch with `setsid` |
| Watcher "started" but died silently | launched with plain `nohup &` → relaunch with `setsid` |
| Launched command appears dead/slow to output | **cold-start, not a hang** — `import pdd.cli` pulls torch/transformers/sae-lens/sklearn ≈ 14s, then loads model+SAE weights before first print. Total silence 20–40s is normal. |
| `nvidia-smi` query empty | check `-i 0` index / driver; fall back to PID-only trigger |
| Two `pgrep` matches | one is a transient — re-run to see the surviving `bash` |

## 5. Environment notes (PDD repo)

- Conda env `pdd`; activate with
  `source /mnt/disk1/miniconda3/etc/profile.d/conda.sh && conda activate pdd`
- PDD CLI: `cd /mnt/disk4/pquan/PDD && python -u -m pdd.cli --config configs/qwen3_1.7b_base.json`
- GPU is a single RTX 4090 (`/dev/nvidia0`, 24 GiB) — shared box, other users'
  jobs run on it. Never kill processes you don't own; chain via the watcher instead.
- Check VRAM before any GPU work:
  `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`
