#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export PYTHON_BIN
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export USE_LORA="${USE_LORA:-true}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export TARGET_MODULES="${TARGET_MODULES:-attn_mlp}"
export REPORT_TO="${REPORT_TO:-none}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

bash "$ROOT_DIR/scripts/run_trl_sft.sh" "$@"
