#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
DEFAULT_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
RANKS="${RANKS:-8,16,32}"
TARGETS="${TARGETS:-attention_only,attn_mlp,all_linear}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/run_trl_lora_ablation.py" \
  --model-path "$MODEL_PATH" \
  --ranks "$RANKS" \
  --target-modules "$TARGETS" \
  "$@"
