#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for the live VideoChat-Flash demo on Jetson
# Captures from /dev/video0 at 10 FPS, batches 100 frames, and streams to the model

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_REPO="OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"
CAMERA_DEVICE="/dev/video0"
FPS="10"
BATCH_SIZE="100"

export HF_HUB_ENABLE_HF_TRANSFER=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec python3 "$SCRIPT_DIR/live_videochat_demo.py" \
  --model "$MODEL_REPO" \
  --camera "$CAMERA_DEVICE" \
  --fps "$FPS" \
  --batch-size "$BATCH_SIZE" \
  --prompt "Describe what happened in the last segment." \
  --keep-history \
  --compress


