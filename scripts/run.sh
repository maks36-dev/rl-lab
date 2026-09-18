#!/usr/bin/env bash
# Запуск прогона на beleriand: свободная GPU → tmux → nice → лог в runs/.
# Использование: scripts/run.sh <chapter_pkg> <run_name> [hydra overrides...]
#   scripts/run.sh ch00_ppo_baseline cheetah_s1 env=halfcheetah seed=1
# GPU: берётся первая FREE из gpu_free.sh, либо заранее выставленный CUDA_VISIBLE_DEVICES.
set -euo pipefail
PKG=$1; NAME=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CHAPTER=${PKG#ch}                              # ch00_ppo_baseline → 00_ppo_baseline
LOG_DIR="runs/$CHAPTER/$NAME"
SESSION="${CHAPTER%%_*}-$NAME"                 # 00-cheetah_s1

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU=$("$ROOT/scripts/gpu_free.sh" | awk '/FREE/ {print $2; exit}')
  [[ -n "$GPU" ]] || { echo "no free GPU (see scripts/gpu_free.sh)"; exit 1; }
else
  GPU=$CUDA_VISIBLE_DEVICES
fi
mkdir -p "$ROOT/$LOG_DIR"
tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session $SESSION already exists"; exit 1; }

CMD="cd '$ROOT' && source scripts/env.sh && export CUDA_VISIBLE_DEVICES=$GPU && \
nice -n 10 python -m chapters.$PKG.src.train run_name=$NAME $* 2>&1 | tee '$LOG_DIR/log.txt'"
tmux new -d -s "$SESSION" "$CMD"
echo "started tmux '$SESSION' on GPU $GPU → $LOG_DIR/log.txt"
echo "follow: tmux attach -t $SESSION   |   tail -f $LOG_DIR/log.txt"
