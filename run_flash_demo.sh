#!/usr/bin/env bash
# run_flash_demo.sh – convenience launcher for videochat_flash_stream.py
#
# Usage:
#   ./run_flash_demo.sh --question "Describe the video." [other options]
# All extra args are passed through to the Python script.

set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------
# Automatic dependency installation
# -----------------------------------------
# Set SKIP_INSTALL=1 in the environment to disable.
# Override index with: PYPI_INDEX_URL=https://your-mirror/simple ./run_flash_demo.sh

if [[ "${SKIP_INSTALL:-0}" -ne 1 ]]; then
  PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
  echo "[INFO] Installing/upgrading required Python packages…"
  python3 -m pip install --upgrade pip
  python3 -m pip install -i "${PYPI_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "transformers==4.40.1" timm av imageio decord opencv-python huggingface_hub

  # flash-attn build can be heavy; try mirror first, fallback to PyPI if missing.
  python3 -m pip install -i "${PYPI_INDEX_URL}" --extra-index-url https://pypi.org/simple \
    flash-attn --no-build-isolation || true
fi

# -----------------------------------------
# Launch demo
# -----------------------------------------

python3 "${SCRIPT_DIR}/videochat_flash_stream.py" --cam /dev/video0 "$@"
