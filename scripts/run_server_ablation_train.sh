#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
mkdir -p "$LOG_DIR"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
RANKS="${RANKS:-8,16,32}"
TARGETS="${TARGETS:-attention_only,attn_mlp,all_linear}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
REPORT_TO="${REPORT_TO:-none}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
RUN_MODE="${RUN_MODE:-foreground}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/ablation_train_${TIMESTAMP}.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/ablation_train.pid}"

CMD=(
  "$ROOT_DIR/scripts/run_trl_lora_ablation_autodl.sh"
  "$@"
)

export PYTHON_BIN
export MODEL_PATH
export RANKS
export TARGETS
export NUM_TRAIN_EPOCHS
export PER_DEVICE_TRAIN_BATCH_SIZE
export PER_DEVICE_EVAL_BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS
export LEARNING_RATE
export MAX_LENGTH
export REPORT_TO
export GRADIENT_CHECKPOINTING

echo "[info] Python: $PYTHON_BIN"
echo "[info] Model: $MODEL_PATH"
echo "[info] Ranks: $RANKS"
echo "[info] Targets: $TARGETS"
echo "[info] Run mode: $RUN_MODE"

if [[ "$RUN_MODE" == "background" ]]; then
  nohup bash "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "[info] Started background ablation training"
  echo "[info] PID: $(cat "$PID_FILE")"
  echo "[info] Log: $LOG_FILE"
else
  exec bash "${CMD[@]}"
fi
