# NanoLLM

Live video chat demos for MacBook Air (MPS) and Jetson Orin Nano (CUDA) using VideoChat-Flash and Qwen2.5-VL models.

## Quick Start

### Jetson Orin Nano
```bash
chmod +x jetson_install_and_run.sh
./jetson_install_and_run.sh
```

### MacBook Air (Apple Silicon)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision transformers==4.40.1 timm imageio opencv-python eva-decord

# For video demos (may have CUDA compatibility issues on Mac)
python live_videochat_demo.py --show-preview --segment-seconds 2 --max-frames 48 --compress

# For image captioning (more stable on Mac)
python live_image_caption_demo.py --show-preview --interval 2
```

## Files

- `jetson_install_and_run.sh` - Complete setup script for Jetson Orin Nano with VideoChat-Flash
- `live_videochat_demo.py` - Continuous camera-to-model demo for MacBook Air (MPS) 
- `live_image_caption_demo.py` - Lightweight image captioning demo using Qwen2.5-VL

## Features

- Handles flash_attn compatibility issues on different platforms
- Uses `/dev/video0` on Jetson, built-in camera on Mac
- Continuous inference loops with configurable FPS and segment length
- Jetson script uses JP6/CUDA 12.6 pip index with PyPI fallback

## Camera Permissions

On macOS, allow camera access in System Settings → Privacy & Security → Camera for Terminal/Python.
