#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_JSONL="${TRAIN_JSONL:-$ROOT_DIR/data/golden_v1_train.jsonl}"
VERL_DATA_DIR="${VERL_DATA_DIR:-$ROOT_DIR/data/verl_sft}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$VERL_DATA_DIR/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$VERL_DATA_DIR/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/verl_sft_run}"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  echo "[error] Please set MODEL_PATH to your base model path or HF repo." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT_DIR/.venv/bin/python" && "${PYTHON_BIN}" == "python3" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
if [[ -x "$ROOT_DIR/.venv/bin/torchrun" && "${TORCHRUN_BIN}" == "torchrun" ]]; then
  TORCHRUN_BIN="$ROOT_DIR/.venv/bin/torchrun"
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
VAL_RATIO="${VAL_RATIO:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
LR="${LR:-2e-5}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.03}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-100}"
PAD_MODE="${PAD_MODE:-no_padding}"
USE_LORA="${USE_LORA:-true}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
TRAINER_DEVICE="${TRAINER_DEVICE:-cpu}"

mkdir -p "$VERL_DATA_DIR" "$OUTPUT_DIR"

echo "[info] Preparing verl SFT parquet data ..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/prepare_verl_sft_dataset.py" \
  --input "$TRAIN_JSONL" \
  --output-dir "$VERL_DATA_DIR" \
  --val-ratio "$VAL_RATIO"

echo "[info] Launching verl SFT training ..."

CMD=(
  "$TORCHRUN_BIN"
  --standalone
  --nnodes=1
  --nproc_per_node="$NPROC_PER_NODE"
  -m
  verl.trainer.sft_trainer
  data.train_files="$TRAIN_PARQUET"
  data.val_files="$VAL_PARQUET"
  data.messages_key=messages
  data.max_length="$MAX_LENGTH"
  data.truncation=error
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.pad_mode="$PAD_MODE"
  model.path="$MODEL_PATH"
  model.use_remove_padding="$USE_REMOVE_PADDING"
  trainer.default_local_dir="$OUTPUT_DIR"
  trainer.project_name=ecom_llm_v1
  trainer.experiment_name=verl_sft_run
  trainer.logger='["console"]'
  trainer.total_epochs="$MAX_EPOCHS"
  trainer.total_training_steps=null
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.device="$TRAINER_DEVICE"
  optim.lr="$LR"
  optim.lr_warmup_steps_ratio="$LR_WARMUP_RATIO"
)

if [[ "$USE_LORA" == "true" ]]; then
  CMD+=(
    model.lora_rank="$LORA_RANK"
    model.lora_alpha="$LORA_ALPHA"
    model.target_modules="$LORA_TARGET_MODULES"
  )
else
  CMD+=(model.lora_rank=0)
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf '[info] Command: '
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
