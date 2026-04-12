#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1090
source "$ROOT_DIR/.venv/bin/activate"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
RANKS="${RANKS:-8,16,32}"
TARGETS="${TARGETS:-attention_only,attn_mlp,all_linear}"

python "$ROOT_DIR/scripts/run_trl_lora_ablation.py" \
  --model-path "$MODEL_PATH" \
  --ranks "$RANKS" \
  --target-modules "$TARGETS" \
  "$@"
