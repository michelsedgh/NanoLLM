### Live VideoChat-Flash Demo (Jetson Orin Nano)

This demo captures 10 FPS from `/dev/video0`, groups 100 frames into a short temporary video, and queries `OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448` in a loop.

#### Quick start

1) Ensure dependencies are installed (Jetson may already have many):

```bash
pip install transformers==4.40.1 timm av imageio decord opencv-python
# optional (if available for your Jetson build)
pip install flash-attn --no-build-isolation
```

2) Run the demo:

```bash
bash jetson_run_demo.sh
```

This will:
- Capture from `/dev/video0` at 10 FPS
- Batch 100 frames per chunk
- Ask: "Describe what happened in the last segment."
- Keep chat history and enable compression

#### Custom usage

```bash
python3 live_videochat_demo.py \
  --model OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448 \
  --camera /dev/video0 \
  --fps 10 \
  --batch-size 100 \
  --prompt "Describe what happened in the last segment." \
  --keep-history \
  --compress \
  --attn sdpa
```

Flags:
- `--camera`: camera device path (default `/dev/video0`)
- `--fps`: capture FPS (default 10)
- `--batch-size`: frames per model call (default 100)
- `--prompt`: text prompt per chunk
- `--keep-history`: maintain conversation between chunks
- `--compress`: enable mm_llm_compress to reduce token load
- `--attn`: attention backend: `sdpa` (default), `eager`, or `flash2` (requires flash-attn)
- `--keep-chunks`: keep temporary chunk files on disk
- `--verbose`: print capture progress

Notes:
- The script enforces 10 FPS by sleeping between captures even if the device FPS differs.
- On Jetson, the script auto-selects a safe dtype (`bfloat16` if supported → `float16` → `float32`).
- If `cv2.VideoWriter` fails for `.mp4`, it will try `.avi` with alternative codecs.
- Flash attention is optional; by default we run with `--attn sdpa` to avoid requiring `flash-attn`.


