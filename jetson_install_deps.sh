#!/usr/bin/env bash
set -euo pipefail

# Installs dependencies for the VideoChat-Flash demo on Jetson
# Uses your requested pip index: pypi.jetson-ai-lab.io/jp6/cu126
# Optional flags:
#   --headless-opencv  -> install opencv-python-headless instead of opencv-python
#   --flash-attn       -> attempt to install flash-attn (optional; may not be available for all Jetson builds)

INDEX_URL_DEFAULT="https://pypi.jetson-ai-lab.io/jp6/cu126"
EXTRA_INDEX_URL_DEFAULT="https://pypi.org/simple"

HEADLESS_OPENCV=0
INSTALL_FLASH_ATTN=0

for arg in "$@"; do
  case "$arg" in
    --headless-opencv)
      HEADLESS_OPENCV=1
      ;;
    --flash-attn)
      INSTALL_FLASH_ATTN=1
      ;;
  esac
done

INDEX_URL="${PIP_INDEX_URL:-$INDEX_URL_DEFAULT}"
EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-$EXTRA_INDEX_URL_DEFAULT}"

echo "Using pip index: $INDEX_URL"
echo "Extra index (fallback): $EXTRA_INDEX_URL"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 not found in PATH" >&2
  exit 1
fi

PIP_CMD=("$PYTHON_BIN" -m pip)

echo "Upgrading pip, setuptools, wheel…"
"${PIP_CMD[@]}" install --upgrade \
  --index-url "$INDEX_URL" --extra-index-url "$EXTRA_INDEX_URL" \
  pip setuptools wheel

echo "Installing core runtime deps (torch + huggingface stack)…"
"${PIP_CMD[@]}" install -U \
  --index-url "$INDEX_URL" --extra-index-url "$EXTRA_INDEX_URL" \
  torch safetensors huggingface_hub accelerate

echo "Installing model and video IO deps…"
"${PIP_CMD[@]}" install -U \
  --index-url "$INDEX_URL" --extra-index-url "$EXTRA_INDEX_URL" \
  "transformers==4.40.1" timm av imageio decord

if [[ "$HEADLESS_OPENCV" -eq 1 ]]; then
  echo "Installing opencv-python-headless…"
  "${PIP_CMD[@]}" install -U \
    --index-url "$INDEX_URL" --extra-index-url "$EXTRA_INDEX_URL" \
    opencv-python-headless
else
  echo "Installing opencv-python…"
  "${PIP_CMD[@]}" install -U \
    --index-url "$INDEX_URL" --extra-index-url "$EXTRA_INDEX_URL" \
    opencv-python
fi

if [[ "$INSTALL_FLASH_ATTN" -eq 1 ]]; then
  echo "Attempting to install flash-attn (optional)…"
  # If this fails on your Jetson build, rerun without --flash-attn
  "${PIP_CMD[@]}" install -U \
    --index-url "$INDEX_URL" --extra-index-url "$EXTRA_INDEX_URL" \
    --no-build-isolation flash-attn || echo "flash-attn install failed (optional)"
fi

echo "Verifying imports…"
"$PYTHON_BIN" - <<'PY'
import sys
ok = True
for name in [
    "torch",
    "transformers",
    "timm",
    "av",
    "imageio",
    "decord",
    "cv2",
    "huggingface_hub",
    "safetensors",
    "accelerate",
]:
    try:
        __import__(name)
        print(f"OK: {name}")
    except Exception as e:
        ok = False
        print(f"FAIL: {name} -> {e}")
if not ok:
    sys.exit(1)
print("All imports succeeded.")
PY

echo "Done. You can now run:"
echo "  bash ./jetson_run_demo.sh"


