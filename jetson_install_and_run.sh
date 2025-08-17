#!/usr/bin/env bash
set -euo pipefail

# Jetson Orin Nano setup + continuous demo runner
# - Installs deps using Jetson AI Lab index (JP6 / CUDA 12.6)
# - Falls back to PyPI for general packages
# - Uses /dev/video0 by default
# - Runs the VideoChat-Flash demo in a loop (capture → save short clip → infer → print)

# ---------- Configurable via env vars or flags ----------
INDEX_URL=${INDEX_URL:-"https://pypi.jetson-ai-lab.io/jp6/cu126"}
EXTRA_INDEX_URL=${EXTRA_INDEX_URL:-"https://pypi.org/simple"}
MODEL_ID=${MODEL_ID:-"OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"}
PROMPT=${PROMPT:-"Describe this video in detail."}
SEGMENT_SECONDS=${SEGMENT_SECONDS:-2}
MAX_FRAMES=${MAX_FRAMES:-64}
TARGET_FPS=${TARGET_FPS:-15}
CAMERA=${CAMERA:-"/dev/video0"}
SHOW_PREVIEW=${SHOW_PREVIEW:-0} # set to 1 to show OpenCV window

# ---------- System prereqs ----------
echo "[+] Installing system packages (sudo required)..."
sudo apt-get update -y
sudo apt-get install -y \
  python3-venv python3-pip \
  python3-opencv \
  v4l-utils \
  ffmpeg \
  git

# ---------- Python environment ----------
echo "[+] Creating venv (with system site packages so python3-opencv is visible)..."
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel

export PIP_INDEX_URL="${INDEX_URL}"
export PIP_EXTRA_INDEX_URL="${EXTRA_INDEX_URL}"

echo "[+] Installing CUDA-enabled torch/vision/audio for JP6/cu126..."
pip install --no-cache-dir torch torchvision torchaudio

echo "[+] Installing model/runtime deps..."
pip install --no-cache-dir "transformers==4.40.1" timm imageio tqdm safetensors huggingface-hub

echo "[+] Installing decoders (prefer decord; fallback to eva-decord). Not installing PyAV to avoid FFmpeg conflicts."
if ! pip install --no-cache-dir decord; then
  pip install --no-cache-dir eva-decord
fi

echo "[+] Installing flash-attn (optional). Will stub if unavailable."
if ! pip install --no-build-isolation --no-cache-dir flash-attn; then
  echo "[-] flash-attn wheel not available; creating stub module so imports succeed."
  mkdir -p flash_attn_stub/flash_attn
  printf "# Minimal stub for flash_attn on Jetson when wheel is unavailable.\n" > flash_attn_stub/flash_attn/__init__.py
  export PYTHONPATH="${PWD}/flash_attn_stub:${PYTHONPATH:-}"
fi

# ---------- Write demo script ----------
cat > jetson_videochat_live.py << 'PY'
import argparse
import os
import time
import tempfile
from datetime import datetime

import cv2
import torch
from transformers import AutoModel, AutoTokenizer


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")


def open_camera(device_path: str, width: int, height: int, target_fps: float) -> cv2.VideoCapture:
    # Prefer V4L2 for /dev/video* devices
    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        # Try default backend
        cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera at {device_path}. Check permissions and that the device exists.")

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if target_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(target_fps))
    return cap


def get_fps(cap: cv2.VideoCapture, default_fps: float = 30.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        if fps is None or float(fps) < 5.0:
            return default_fps
        return float(fps)
    except Exception:
        return default_fps


def write_segment_avi(frames, fps: float, frame_size: tuple[int, int]) -> str:
    # Use MJPG AVI for broad compatibility with decoders on Jetson
    tmp_dir = tempfile.gettempdir()
    avi_path = os.path.join(tmp_dir, f"vcf_segment_{int(time.time()*1000)}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(avi_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError("Failed to initialize MJPG video writer.")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return avi_path


def load_model(model_id: str):
    print(f"Loading model {model_id} on cuda with fp16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(dtype=torch.float16, device="cuda")  # type: ignore[attr-defined]
    model.eval()
    try:
        _ = model.get_vision_tower().image_processor  # type: ignore[attr-defined]
    except Exception:
        pass
    # Enable compression for speed by default on Jetson
    if getattr(model, "config", None) is not None:
        model.config.mm_llm_compress = True  # type: ignore[attr-defined]
        model.config.llm_compress_type = "uniform0_attention"  # type: ignore[attr-defined]
        model.config.llm_compress_layer_list = [4, 18]  # type: ignore[attr-defined]
        model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]  # type: ignore[attr-defined]
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Jetson continuous camera → VideoChat-Flash demo")
    parser.add_argument("--prompt", type=str, default=os.environ.get("PROMPT", "Describe this video in detail."))
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL_ID", "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"))
    parser.add_argument("--segment-seconds", type=float, default=float(os.environ.get("SEGMENT_SECONDS", 2)))
    parser.add_argument("--max-frames", type=int, default=int(os.environ.get("MAX_FRAMES", 64)))
    parser.add_argument("--camera", type=str, default=os.environ.get("CAMERA", "/dev/video0"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target-fps", type=float, default=float(os.environ.get("TARGET_FPS", 15)))
    parser.add_argument("--show-preview", dest="show_preview", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=192,
        top_p=0.1,
        num_beams=1,
    )

    # Open camera; if a path like /dev/video0, pass directly, else try index
    cam_arg = args.camera
    if cam_arg.isdigit():
        cam_arg = int(cam_arg)
    cap = open_camera(cam_arg, args.width, args.height, args.target_fps)
    camera_fps = get_fps(cap)
    fps = args.target_fps if camera_fps < 5.0 else min(camera_fps, args.target_fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height or 480)

    print(f"Camera opened: {args.camera} {frame_width}x{frame_height} @ target ~{fps:.1f} FPS (camera reports ~{camera_fps:.1f})")
    print("Press Ctrl+C to stop. Running continuous segments...\n")

    try:
        while True:
            frames = []
            start_time = time.time()
            next_refresh = start_time
            while (time.time() - start_time) < args.segment_seconds:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                frames.append(frame)

                if args.show_preview:
                    preview = frame.copy()
                    cv2.putText(preview, "Recording...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow("VideoChat-Flash Preview", preview)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

                now = time.time()
                if now < next_refresh:
                    time.sleep(min(0.01, next_refresh - now))
                next_refresh = now + (1.0 / max(fps, 1.0))

            if len(frames) == 0:
                print("No frames captured; retrying...")
                continue

            if len(frames) > args.max_frames:
                stride = max(1, len(frames) // args.max_frames)
                frames = frames[::stride][: args.max_frames]

            video_path = write_segment_avi(frames, fps=fps, frame_size=(frame_width, frame_height))

            try:
                t0 = time.time()
                output, _history = model.chat(
                    video_path=video_path,
                    tokenizer=tokenizer,
                    user_prompt=args.prompt,
                    return_history=True,
                    max_num_frames=min(args.max_frames, len(frames)),
                    generation_config=generation_config,
                )
                dt = time.time() - t0
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Output ({dt:.2f}s):\n{output}\n")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Inference error: {e}")
            finally:
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if args.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
PY

# ---------- Run demo ----------
RUNTIME_FLAGS=(
  --prompt "${PROMPT}"
  --model "${MODEL_ID}"
  --segment-seconds "${SEGMENT_SECONDS}"
  --max-frames "${MAX_FRAMES}"
  --camera "${CAMERA}"
  --target-fps "${TARGET_FPS}"
)

if [[ "${SHOW_PREVIEW}" == "1" ]]; then
  RUNTIME_FLAGS+=(--show-preview)
fi

echo "[+] Starting continuous demo... (Ctrl+C to stop)"
python jetson_videochat_live.py "${RUNTIME_FLAGS[@]}"


