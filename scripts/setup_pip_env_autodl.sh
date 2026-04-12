#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TORCH="${INSTALL_TORCH:-false}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Python not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "[info] Python: $(command -v "$PYTHON_BIN")"
echo "[info] Python version: $($PYTHON_BIN --version 2>&1)"
echo "[info] pip: $($PYTHON_BIN -m pip --version 2>&1)"

echo "[info] Upgrading pip/setuptools/wheel in current Python environment"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

if [[ "$INSTALL_TORCH" == "true" ]]; then
  echo "[info] Installing torch stack from $TORCH_INDEX_URL"
  "$PYTHON_BIN" -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "$TORCH_INDEX_URL"
else
  echo "[info] Skipping torch installation (INSTALL_TORCH=false)."
  echo "[info] This assumes your current AutoDL Python environment already has torch."
fi

echo "[info] Installing project training dependencies into current Python environment"
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements-train-autodl.txt"

echo "[info] Environment ready"
"$PYTHON_BIN" - <<'PY'
import sys
print('python_executable', sys.executable)
try:
    import torch
    print('torch', torch.__version__)
    print('cuda_available', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device_count', torch.cuda.device_count())
except Exception as exc:
    print('torch_check_failed', repr(exc))
PY
