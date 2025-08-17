#!/usr/bin/env python3
"""
videochat_flash_stream.py

A lightweight terminal application for real-time video querying with
OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448 on NVIDIA Jetson (Orin Nano).

Features
--------
1. Captures webcam at ~30-40 FPS using OpenCV.
2. Keeps 1 frame every `--sample-every` raw frames (default 10 → ≈3 FPS).
3. Maintains a sliding window of `--window` frames (default 39 ≈13 s).
4. Runs model inference **asynchronously** so capture never stalls.
5. Prints answers to the terminal; extend as desired for further logic.

Install deps
------------
python3 -m pip install --upgrade "transformers==4.40.1" timm av imageio decord opencv-python
# (optional, but recommended for Qwen2) flash-attn --no-build-isolation

Usage
-----
python3 videochat_flash_stream.py --question "Describe this video."

Press Ctrl+C to quit.
"""

import argparse
import os
import threading
import time
import tempfile
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "OpenGVLab/VideoChat-Flash-Qwen2_5-2B_res448"


def frames_to_video(frames: List, path: str, fps: int = 3) -> None:
    """Dump a list of BGR frames to *path* (MP4)."""
    if not frames:
        raise ValueError("No frames provided to frames_to_video().")
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def load_model(device: str = "cuda"):
    print("[INFO] Loading tokenizer and model …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    try:
        model = (
            AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
            .to(torch.bfloat16)
            .to(device)
        )
    except (RuntimeError, ValueError):
        # Fallback to fp16 if bfloat16 not supported on Jetson (likely case).
        print("[WARN] bfloat16 not supported – falling back to fp16")
        model = (
            AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
            .half()
            .to(device)
        )

    model.eval()
    # Disable compression by default.
    model.config.mm_llm_compress = False
    print("[INFO] Model loaded ✔")
    return model, tokenizer


def run_inference(
    frames: List,
    model,
    tokenizer,
    question: str,
    max_num_frames: int,
    generation_config: dict,
):
    """Blocking helper executed in a background thread."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        frames_to_video(frames, tmp_path, fps=3)
        output, _ = model.chat(
            video_path=tmp_path,
            tokenizer=tokenizer,
            user_prompt=question,
            return_history=False,
            max_num_frames=max_num_frames,
            generation_config=generation_config,
        )
        print("\n[RESULT]", output, "\n")
    except Exception as e:
        print("[ERROR] Inference failed:", e)
    finally:
        os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Realtime VideoChat-Flash demo")
    parser.add_argument(
        "--cam",
        type=str,
        default="/dev/video0",
        help="Camera device (index or path, e.g. 0 or /dev/video0)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Keep 1 frame every N raw frames (≈FPS/ N)",
    )
    parser.add_argument(
        "--window", type=int, default=39, help="Sliding window size (kept frames)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe this video in detail.",
        help="Prompt asked each cycle",
    )
    args = parser.parse_args()

    # Model setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device)

    generation_config = dict(
        do_sample=False,
        temperature=0.0,
        max_new_tokens=1024,
        top_p=0.1,
        num_beams=1,
    )

    cam_arg = int(args.cam) if str(args.cam).isdigit() else args.cam
    cap = cv2.VideoCapture(cam_arg)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.cam}")

    kept: Deque = deque(maxlen=args.window)
    raw_idx = 0
    infer_thread: Optional[threading.Thread] = None

    print("[INFO] Starting capture – press Ctrl+C to exit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed, retrying …")
                time.sleep(0.01)
                continue

            raw_idx += 1
            if raw_idx % args.sample_every == 0:
                kept.append(frame.copy())

            # When we have a full buffer and no inference running, trigger one.
            if (
                len(kept) == args.window
                and (infer_thread is None or not infer_thread.is_alive())
            ):
                frames_batch = list(kept)
                infer_thread = threading.Thread(
                    target=run_inference,
                    args=(
                        frames_batch,
                        model,
                        tokenizer,
                        args.question,
                        args.window,
                        generation_config,
                    ),
                    daemon=True,
                )
                infer_thread.start()

            # Brief sleep to avoid burning CPU when capture is very fast.
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n[INFO] Exiting …")
    finally:
        cap.release()
        if infer_thread is not None:
            infer_thread.join()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
