#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-sshleifer/tiny-gpt2}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/trl_sft_smoke}"
export MAX_LENGTH="${MAX_LENGTH:-128}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-1}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export USE_LORA="${USE_LORA:-true}"
export LORA_R="${LORA_R:-8}"
export LORA_ALPHA="${LORA_ALPHA:-16}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export TARGET_MODULES="${TARGET_MODULES:-all_linear}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
export REPORT_TO="${REPORT_TO:-none}"

bash "$ROOT_DIR/scripts/run_trl_sft.sh" "$@"
