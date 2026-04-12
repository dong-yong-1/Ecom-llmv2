#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/data/golden_v1_train.jsonl}"
TRL_DATA_DIR="${TRL_DATA_DIR:-$ROOT_DIR/data/trl_sft}"
TRAIN_FILE="${TRAIN_FILE:-$TRL_DATA_DIR/train.jsonl}"
VAL_FILE="${VAL_FILE:-$TRL_DATA_DIR/val.jsonl}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/trl_sft_run}"
VAL_RATIO="${VAL_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
USE_LORA="${USE_LORA:-true}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-attn_mlp}"
REPORT_TO="${REPORT_TO:-none}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
BF16="${BF16:-false}"
FP16="${FP16:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "[error] Please set MODEL_PATH to your base model path or HF repo." >&2
  exit 1
fi

mkdir -p "$TRL_DATA_DIR" "$OUTPUT_DIR"

echo "[info] Python: $(command -v "$PYTHON_BIN")"
echo "[info] Preparing TRL SFT data ..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_trl_sft_dataset.py" \
  --input "$TRAIN_JSONL" \
  --output-dir "$TRL_DATA_DIR" \
  --val-ratio "$VAL_RATIO"

CMD=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/run_trl_sft.py"
  --model-name-or-path "$MODEL_PATH"
  --train-file "$TRAIN_FILE"
  --eval-file "$VAL_FILE"
  --output-dir "$OUTPUT_DIR"
  --max-length "$MAX_LENGTH"
  --learning-rate "$LEARNING_RATE"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
  --max-steps "$MAX_STEPS"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --warmup-ratio "$WARMUP_RATIO"
  --logging-steps "$LOGGING_STEPS"
  --eval-steps "$EVAL_STEPS"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --eval-strategy "$EVAL_STRATEGY"
  --save-strategy "$SAVE_STRATEGY"
  --target-modules "$TARGET_MODULES"
  --report-to "$REPORT_TO"
)

if [[ "$USE_LORA" == "true" ]]; then
  CMD+=(--use-lora --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT")
else
  CMD+=(--no-lora)
fi

if [[ "$GRADIENT_CHECKPOINTING" == "true" ]]; then
  CMD+=(--gradient-checkpointing)
fi
if [[ "$BF16" == "true" ]]; then
  CMD+=(--bf16)
fi
if [[ "$FP16" == "true" ]]; then
  CMD+=(--fp16)
fi
if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
  CMD+=(--trust-remote-code)
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf '[info] Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
