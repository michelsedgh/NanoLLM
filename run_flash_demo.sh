#!/usr/bin/env bash
# run_flash_demo.sh – convenience launcher for videochat_flash_stream.py
#
# Features:
#   1. Optionally installs dependencies from the Jetson AI Lab mirror
#      (pypi.jetson-ai-lab.io/jp6/cu126) unless SKIP_INSTALL=1.
#   2. Forwards all CLI args to the Python demo.
#   3. Defaults camera to /dev/video0.
#
# Usage examples:
#   ./run_flash_demo.sh --question "Describe the video."
#   SKIP_INSTALL=1 ./run_flash_demo.sh --sample-every 8
#   PYPI_INDEX_URL=https://your-mirror/simple ./run_flash_demo.sh --window 50
#
set -euo pipefail

# Directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#############################################
# Dependency installation (optional)
#############################################
if [[ "${SKIP_INSTALL:-0}" -ne 1 ]]; then
  PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
  echo "[INFO] Installing/upgrading required Python packages via ${PYPI_INDEX_URL} …"
  python3 -m pip install --upgrade pip
  python3 -m pip install -i "${PYPI_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "transformers==4.40.1" timm av imageio decord opencv-python huggingface_hub

  # flash-attn is optional but recommended; ignore failure if wheel not present.
  python3 -m pip install -i "${PYPI_INDEX_URL}" --extra-index-url https://pypi.org/simple \
    flash-attn --no-build-isolation || true
fi

#############################################
# Launch the demo
#############################################
python3 "${SCRIPT_DIR}/videochat_flash_stream.py" --cam /dev/video0 "$@"
