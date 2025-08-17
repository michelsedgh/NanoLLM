#!/usr/bin/env bash
set -euo pipefail

# Simple runner for Jetson Orin Nano after manual setup
# Assumes you have already installed dependencies manually

# Default settings for Jetson
MODEL_ID=${MODEL_ID:-"OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"}
PROMPT=${PROMPT:-"Describe this video in detail."}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-2}
MAX_FRAMES=${MAX_FRAMES:-64}
TARGET_FPS=${TARGET_FPS:-15}
CAMERA=${CAMERA:-"/dev/video0"}
SHOW_PREVIEW=${SHOW_PREVIEW:-0}

# Check if camera exists
if [[ ! -e "${CAMERA}" ]]; then
    echo "Error: Camera ${CAMERA} not found. Available video devices:"
    ls /dev/video* 2>/dev/null || echo "No video devices found"
    exit 1
fi

# Check if we have the demo script
if [[ ! -f "live_videochat_demo.py" ]]; then
    echo "Error: live_videochat_demo.py not found in current directory"
    echo "Make sure you're running this from the directory containing the demo script"
    exit 1
fi

# Setup environment for CUDA on Jetson
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Create flash_attn stub if needed (some models hard-require it for imports)
if ! python3 -c "import flash_attn" 2>/dev/null; then
    echo "[INFO] flash_attn not available, creating stub for compatibility..."
    mkdir -p flash_attn_stub/flash_attn
    echo "# Stub for flash_attn compatibility" > flash_attn_stub/flash_attn/__init__.py
    export PYTHONPATH="${PWD}/flash_attn_stub:${PYTHONPATH:-}"
fi

# Build command arguments
ARGS=(
    --prompt "${PROMPT}"
    --model "${MODEL_ID}"
    --segment-seconds "${SEGMENT_SECONDS}"
    --max-frames "${MAX_FRAMES}"
    --camera "${CAMERA}"
    --target-fps "${TARGET_FPS}"
    --width 640
    --height 480
    --compress
)

if [[ "${SHOW_PREVIEW}" == "1" ]]; then
    ARGS+=(--show-preview)
fi

echo "[+] Starting Jetson video chat demo..."
echo "    Model: ${MODEL_ID}"
echo "    Camera: ${CAMERA}"
echo "    Segment: ${SEGMENT_SECONDS}s, ${MAX_FRAMES} frames max"
echo "    FPS: ${TARGET_FPS}"
echo "    Preview: $([[ "${SHOW_PREVIEW}" == "1" ]] && echo "enabled" || echo "disabled")"
echo "    Prompt: ${PROMPT}"
echo ""
echo "Press Ctrl+C to stop..."

# Run the demo with device path as camera arg
python3 live_videochat_demo.py "${ARGS[@]}"
