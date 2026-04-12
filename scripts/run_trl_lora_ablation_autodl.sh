#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
RANKS="${RANKS:-8,16,32}"
TARGETS="${TARGETS:-attention_only,attn_mlp,all_linear}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/run_trl_lora_ablation.py" \
  --model-path "$MODEL_PATH" \
  --ranks "$RANKS" \
  --target-modules "$TARGETS" \
  "$@"
