#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] uv is not installed. Please install uv first on AutoDL." >&2
  echo "[hint] curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "[info] Creating uv environment at $VENV_DIR"
uv venv --python "$PYTHON_BIN" "$VENV_DIR"

echo "[info] Activating environment"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[info] Installing training dependencies"
uv pip install -r "$ROOT_DIR/requirements-train.txt"

echo "[info] Done. Activate with: source $VENV_DIR/bin/activate"
