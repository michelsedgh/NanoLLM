import argparse
import os
import sys
import time
import tempfile
import shutil
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import torch
from transformers import AutoModel, AutoTokenizer


def select_best_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def human_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def open_camera(camera_path: str, width: Optional[int], height: Optional[int], requested_fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {camera_path}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    # Best-effort: some devices ignore this; we will also sleep to enforce 10 FPS
    cap.set(cv2.CAP_PROP_FPS, float(requested_fps))
    return cap


def capture_frame_batch(
    cap: cv2.VideoCapture,
    batch_size: int,
    fps_limit: int,
    verbose: bool = False,
) -> List["np.ndarray"]:
    import numpy as np  # Lazy import

    frames: List[np.ndarray] = []
    frame_interval_s = 1.0 / float(fps_limit)
    next_capture_time = time.perf_counter()

    while len(frames) < batch_size:
        now = time.perf_counter()
        if now < next_capture_time:
            # Sleep minimally to hit 10 FPS
            time.sleep(max(0.0, next_capture_time - now))
        next_capture_time = time.perf_counter() + frame_interval_s

        ok, frame = cap.read()
        if not ok or frame is None:
            # Brief backoff if the camera skipped a frame
            time.sleep(0.01)
            continue

        frames.append(frame)
        if verbose and len(frames) % 10 == 0:
            print(f"[{human_time()}] captured {len(frames)} / {batch_size} frames", flush=True)

    return frames


def try_open_videowriter(
    output_path_base: str,
    width: int,
    height: int,
    fps: int,
) -> Tuple[Optional[cv2.VideoWriter], Optional[str]]:
    attempts = [
        ("mp4", "mp4v"),
        ("mp4", "avc1"),
        ("avi", "XVID"),
        ("avi", "MJPG"),
    ]

    for ext, fourcc_str in attempts:
        out_path = f"{output_path_base}.{ext}"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (width, height))
        if writer is not None and writer.isOpened():
            return writer, out_path
        # Cleanup failed attempt
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
    return None, None


def write_video_chunk(frames: List["np.ndarray"], fps: int, tmp_dir: str) -> str:
    import numpy as np  # Lazy import

    if len(frames) == 0:
        raise ValueError("No frames to write")

    height, width = frames[0].shape[:2]
    # Ensure all frames have identical size
    normalized_frames: List[np.ndarray] = []
    for f in frames:
        if f.shape[0] != height or f.shape[1] != width:
            f = cv2.resize(f, (width, height), interpolation=cv2.INTER_LINEAR)
        normalized_frames.append(f)

    base = os.path.join(tmp_dir, f"video_chunk_{int(time.time())}")
    writer, path = try_open_videowriter(base, width, height, fps)
    if writer is None or path is None:
        raise RuntimeError("Could not open a suitable VideoWriter. Install ffmpeg/gstreamer support or try a different codec.")

    for f in normalized_frames:
        writer.write(f)
    writer.release()
    return path


def _set_torch_sdp_policy(attn: str) -> None:
    # Prefer non-flash attention by default to avoid hard dependency on flash-attn
    try:
        from torch.backends.cuda import sdp_kernel
        if attn == "flash2":
            sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        elif attn == "eager":
            # Disable flash and mem-efficient kernels to force math (eager) path
            sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        else:  # sdpa
            # Disable flash kernel; allow math path
            sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except Exception:
        # Older Torch versions may not have sdp_kernel; ignore
        pass


def load_model_and_tokenizer(model_path: str, dtype: torch.dtype, device: str, attn: str):
    print(f"[{human_time()}] Loading model: {model_path}", flush=True)
    _set_torch_sdp_policy(attn)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    attn_impl_map = {"sdpa": "sdpa", "eager": "eager", "flash2": "flash_attention_2"}
    attn_impl = attn_impl_map.get(attn, "sdpa")

    try:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    except TypeError:
        # attn_implementation not supported by this model class
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        try:
            model.config.attn_implementation = attn_impl
        except Exception:
            pass

    model = model.to(dtype).to(device).eval()
    return model, tokenizer


def run_loop(
    model,
    tokenizer,
    camera_path: str,
    batch_size: int,
    fps: int,
    prompt: str,
    keep_history: bool,
    mm_llm_compress: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
    device: str,
    keep_chunks: bool,
    verbose: bool,
):
    # Optional compression settings from model card
    if mm_llm_compress:
        model.config.mm_llm_compress = True
        model.config.llm_compress_type = "uniform0_attention"
        model.config.llm_compress_layer_list = [4, 18]
        model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
    else:
        model.config.mm_llm_compress = False

    generation_config = dict(
        do_sample=temperature > 0.0,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        num_beams=num_beams,
    )

    cap = open_camera(camera_path, width=None, height=None, requested_fps=fps)

    tmp_dir = tempfile.mkdtemp(prefix="videochat_flash_")
    print(f"[{human_time()}] Temporary directory: {tmp_dir}", flush=True)

    chat_history = None
    chunk_idx = 0

    try:
        while True:
            chunk_idx += 1
            print(f"[{human_time()}] Capturing chunk #{chunk_idx} ({batch_size} frames @ {fps} FPS)…", flush=True)

            frames = capture_frame_batch(cap, batch_size=batch_size, fps_limit=fps, verbose=verbose)

            video_path = write_video_chunk(frames, fps=fps, tmp_dir=tmp_dir)
            print(f"[{human_time()}] Wrote video chunk: {video_path}", flush=True)

            with torch.inference_mode():
                if keep_history and chat_history is not None:
                    output, chat_history = model.chat(
                        video_path=video_path,
                        tokenizer=tokenizer,
                        user_prompt=prompt,
                        chat_history=chat_history,
                        return_history=True,
                        max_num_frames=batch_size,
                        generation_config=generation_config,
                    )
                else:
                    output, chat_history = model.chat(
                        video_path=video_path,
                        tokenizer=tokenizer,
                        user_prompt=prompt,
                        return_history=True,
                        max_num_frames=batch_size,
                        generation_config=generation_config,
                    )

            print()
            print("=" * 80)
            print(f"Chunk #{chunk_idx} @ {human_time()}:")
            print(output)
            print("=" * 80)
            print()

            if not keep_chunks:
                try:
                    os.remove(video_path)
                except Exception:
                    pass

            # Free some memory in tight VRAM environments
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print(f"\n[{human_time()}] Interrupted by user. Shutting down…", flush=True)
    finally:
        cap.release()
        if not keep_chunks:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Live VideoChat-Flash demo from /dev/video0 with 10 FPS and 100-frame chunks.")
    parser.add_argument("--model", default="OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448", help="Model repo or local path")
    parser.add_argument("--camera", default="/dev/video0", help="Video device path")
    parser.add_argument("--fps", type=int, default=10, help="Capture FPS")
    parser.add_argument("--batch-size", type=int, default=100, help="Frames per chunk")
    parser.add_argument("--prompt", default="Describe what happened in the last segment.", help="Prompt for each chunk")
    parser.add_argument("--compress", action="store_true", help="Enable mm_llm_compress settings to reduce token load")
    parser.add_argument("--keep-history", action="store_true", help="Preserve chat history across chunks")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 disables sampling)")
    parser.add_argument("--top-p", type=float, default=0.1, help="Top-p sampling")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam search beams")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager", "flash2"], help="Attention backend (default sdpa; avoids flash-attn dependency)")
    parser.add_argument("--keep-chunks", action="store_true", help="Do not delete the temporary video chunks")
    parser.add_argument("--verbose", action="store_true", help="Verbose capture logs")
    args = parser.parse_args()

    # Select dtype suitable for Jetson Orin Nano
    dtype = select_best_dtype()
    print(f"[{human_time()}] Using device={args.device}, dtype={dtype}")

    model, tokenizer = load_model_and_tokenizer(args.model, dtype, args.device, args.attn)

    # Touch the vision tower to ensure image processor is registered; not directly used here
    try:
        _ = model.get_vision_tower().image_processor
    except Exception:
        pass

    run_loop(
        model=model,
        tokenizer=tokenizer,
        camera_path=args.camera,
        batch_size=args.batch_size,
        fps=args.fps,
        prompt=args.prompt,
        keep_history=args.keep_history,
        mm_llm_compress=args.compress,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        device=args.device,
        keep_chunks=args.keep_chunks,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()


