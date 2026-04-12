#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
INSTALL_TORCH="${INSTALL_TORCH:-true}"
RECREATE_VENV="${RECREATE_VENV:-false}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[error] uv is not installed. Please install uv first." >&2
  exit 1
fi

if [[ -d "$VENV_DIR" && "$RECREATE_VENV" != "true" ]]; then
  echo "[info] Reusing existing uv environment at $VENV_DIR"
else
  echo "[info] Creating uv environment at $VENV_DIR"
  if [[ "$RECREATE_VENV" == "true" ]]; then
    uv venv --clear --python "$PYTHON_BIN" "$VENV_DIR"
  else
    uv venv --python "$PYTHON_BIN" "$VENV_DIR"
  fi
fi

echo "[info] Installing base training dependencies"
uv pip install --python "$VENV_DIR/bin/python" -r "$ROOT_DIR/requirements-train.txt"

if [[ "$INSTALL_TORCH" == "true" ]]; then
  echo "[info] Installing local torch stack for macOS / CPU / MPS"
  uv pip install --python "$VENV_DIR/bin/python" torch torchvision torchaudio
else
  echo "[info] Skipping torch installation (INSTALL_TORCH=false)"
fi

echo "[info] Environment ready"
"$VENV_DIR/bin/python" - <<'PY'
import sys
print("python_executable", sys.executable)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    print("mps_available", mps_available)
except Exception as exc:
    print("torch_check_failed", repr(exc))
PY

echo "[info] Activate with: source $VENV_DIR/bin/activate"
